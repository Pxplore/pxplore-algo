import sys
import os
import time
import pika
from pymongo import MongoClient
from bson import ObjectId
import httpx
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime._exceptions import ArkAPIError
from retry import retry
from config import LLM, MONGO
from utils import get_channel, now, get_logger
from service.llm.base import BASE_LLM_CACHE, NO_CACHE_YET
import json
import ast


class VOLCARK(BASE_LLM_CACHE):
	"""
	A class to interact with VolcArk's language model with an optional cache mechanism.
	Inherits from BASE_LLM_CACHE.
	"""
	def __init__(self, api_key=None, cache_collection=None, **kwargs):
		"""
		Initializes the VOLCARK class.

		Args:
			api_key (str, optional): The API key to authenticate with VolcArk. 
				Alternatively, use environment variables or IAM credentials as described in VolcArk docs.
			cache_collection (MongoDB collection, optional): The cache collection to store requests/responses.
			**kwargs: Additional keyword arguments for the Ark client (e.g., ak, sk, etc.).
		"""
		# For example, if you want to pass AK/SK or different arguments, do so here.
		# If using API key directly:
		#   self.llm_client = Ark(api_key=api_key, timeout=httpx.Timeout(timeout=1800), **kwargs)
		# Or if using IAM credentials from environment, skip api_key:
		#   self.llm_client = Ark(timeout=httpx.Timeout(timeout=1800), **kwargs)
		self.llm_client = Ark(
			api_key=api_key, 
			timeout=httpx.Timeout(timeout=1800),
			max_output_tokens=32768,
			**kwargs
		)
		self.cost = dict()
		self.cache_collection = cache_collection
		if not self.cache_collection:
			self.cache_collection = MongoClient(
				MONGO.HOST,
				MONGO.PORT
			).llm.volcark_cache

	@retry(ArkAPIError, tries=3)
	def _call_model(self, query: dict, use_cache):
		"""
		Calls the VolcArk model with the given query.

		Args:
			query (dict): The query to send to the VolcArk model. Should match the
						parameters for `client.chat.completions.create(...)`.
			use_cache (bool): Whether to use cached responses if available.

		Returns:
			str: The response from the model (combined reasoning + content), 
				either from cache or via a new request.
		"""
		assert (query.get("stream", False) == False), "Don't make a stream request via non-stream calling"
		response = NO_CACHE_YET
		# 1. Check cache (if enabled)
		if use_cache:
			response = self.check_cache(query)

		# 2. If not in cache, call model
		if response == NO_CACHE_YET:
			# If you're using streaming in production, you'll need to accumulate chunks.
			# For simplicity here, we'll do a direct non-streaming call.
			temp_query = query.copy()
			temp_query["model"] = LLM.VOLCARK.MODEL_NAME_MAPPING[query["model"]]

			resp = self.llm_client.chat.completions.create(**temp_query)

			
			reasoning = resp.choices[0].message.reasoning_content or None
			response = resp.choices[0].message.content or ""

			# If VolcArk returns usage info, you can track cost here. 
			# (The example does not show usage, so we skip.)
			usage = resp.usage.model_dump()
			# 3. Write to cache
			self.write_cache(query, response, usage=usage, reasoning=reasoning)  # Or store usage if available
			if reasoning:
				response = [response, reasoning]

		return response


class VOLCARK_SERVICE:
	"""
	A service class for managing VolcArk LLM requests using MongoDB and RabbitMQ.
	"""
	collection = MongoClient(
		MONGO.HOST,
		MONGO.PORT
	).llm.volcark  # Where main job records are stored
	queue_name = "llm-volcark"

	logger = get_logger(__name__, __file__)

	@staticmethod
	def trigger(
		parent_service: str,
		parent_job_id=None,  # set to ObjectId if you want callback
		use_cache=False,
		**query
	) -> str:
		"""
		Creates and triggers a new job for a VolcArk LLM request via RabbitMQ.

		Args:
			parent_service (str): The service initiating this request (used for callback).
			parent_job_id (ObjectId, optional): Parent job ID if a callback is needed.
			use_cache (bool, optional): Whether to use cached responses.
			**query: The query parameters for the VolcArk model.
		
		Returns:
			str: The job ID of the created request in MongoDB.
		"""
		connection = pika.BlockingConnection(
			pika.ConnectionParameters(host='localhost')
		)
		channel = connection.channel()
		channel.queue_declare(queue=VOLCARK_SERVICE.queue_name, durable=True)

		VOLCARK_SERVICE.logger.info("Writing job to Mongo")
		job_id = VOLCARK_SERVICE.collection.insert_one(
			dict(
				parent_service=parent_service,
				parent_job_id=parent_job_id,
				created_time=now(),
				use_cache=use_cache,
				query=query,
			)
		).inserted_id

		VOLCARK_SERVICE.logger.info("Pushing job to RabbitMQ")
		channel.basic_publish(
			exchange="",
			routing_key=VOLCARK_SERVICE.queue_name,
			body=str(job_id)
		)
		connection.close()

		VOLCARK_SERVICE.logger.info("Job pushed to RabbitMQ")
		return job_id

	@staticmethod
	def launch_worker():
		"""
		Launches a worker that processes jobs from the RabbitMQ queue.
		The worker uses the VOLCARK class to request completions and
		stores responses back to MongoDB.
		"""
		try:
			connection = pika.BlockingConnection(
				pika.ConnectionParameters(host='localhost')
			)
			channel = connection.channel()

			channel.queue_declare(
				queue=VOLCARK_SERVICE.queue_name,
				durable=True
			)

			# Initialize the VolcArk client
			llm_controller = VOLCARK(
				api_key=LLM.VOLCARK.API_KEY, 
				# or ak=..., sk=..., etc. if you're using IAM
			)

			def callback(ch, method, properties, body):
				job_id = ObjectId(body.decode())
				VOLCARK_SERVICE.logger.info(f"Received VolcArk LLM job - {job_id}")
				job = VOLCARK_SERVICE.collection.find_one(dict(_id=job_id))
				query = job["query"]
				use_cache = job["use_cache"]
				
				parent_service = job["parent_service"]
				parent_job_id = job["parent_job_id"]

				VOLCARK_SERVICE.logger.debug(f"VolcArk LLM Query - {query}")

				# Call the model
				ret = llm_controller._call_model(
					query=query,
					use_cache=use_cache
				)

				# Update the Mongo record
				VOLCARK_SERVICE.collection.update_one(
					dict(_id=job_id),
					{"$set": dict(
						completion_time=now(),
						response=ret
					)}
				)
				VOLCARK_SERVICE.logger.debug(f"VolcArk LLM Output - {ret}")

				# If a parent job needs a callback, push it to that queue
				if parent_job_id:
					parent_connection, parent_channel = get_channel(parent_service)
					parent_channel.basic_publish(
						exchange="",
						routing_key=parent_service,
						body=str(parent_job_id)
					)
					parent_connection.close()

				ch.basic_ack(delivery_tag=method.delivery_tag)

			channel.basic_consume(
				queue=VOLCARK_SERVICE.queue_name,
				on_message_callback=callback,
				auto_ack=False,
			)
			VOLCARK_SERVICE.logger.info('VolcArk Worker Launched. To exit press CTRL+C')
			channel.start_consuming()

		except KeyboardInterrupt:
			VOLCARK_SERVICE.logger.warning('Shutting Off Worker')
			try:
				sys.exit(0)
			except SystemExit:
				os._exit(0)

	@staticmethod
	def get_response(job_id):
		"""
		Retrieves the response of a job with the given ID from MongoDB.

		Args:
			job_id (ObjectId): The ID of the job to retrieve.

		Returns:
			str: The model response, or None if not found or not finished.
		"""
		record = VOLCARK_SERVICE.collection.find_one(dict(_id=job_id))
		if not record:
			VOLCARK_SERVICE.logger.error(f"Job With ID of {job_id} not found")
		elif "completion_time" not in record:
			VOLCARK_SERVICE.logger.error(f"Job {job_id} has not completed yet.")
		else:
			return record["response"]
		return None

	@staticmethod
	def get_response_sync(job_id, timeout=300):
		"""
		Retrieves the response of a job with the given ID synchronously, waiting up to `timeout` seconds.

		Args:
			job_id (ObjectId): The ID of the job to retrieve.
			timeout (int, optional): Maximum time to wait in seconds. Defaults to 300.

		Returns:
			str: The model response, or None if not found or timed out.
		"""
		start_time = time.time()
		while (time.time() - start_time) < timeout:
			record = VOLCARK_SERVICE.collection.find_one(dict(_id=job_id))
			if not record:
				VOLCARK_SERVICE.logger.error(f"Job With ID of {job_id} not found")
				return None
			elif "completion_time" in record:
				return record["response"]
			time.sleep(1)  # Wait 1 second between checks

		VOLCARK_SERVICE.logger.error(f"Retrieving Response From Job {job_id} Timed Out After {timeout} Seconds")
		return None

if __name__ == "__main__":
	VOLCARK_SERVICE.logger.warning("STARTING VOLCARK LLM SERVICE")
	VOLCARK_SERVICE.launch_worker()