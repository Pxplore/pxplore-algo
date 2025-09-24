import asyncio
import sys
import os
import time
from anthropic import Anthropic, RateLimitError
import pika
from pymongo import MongoClient
from bson import ObjectId
from retry import retry
from config import LLM, MONGO
from utils import get_channel, now, get_logger
from service.llm.base import BASE_LLM_CACHE, NO_CACHE_YET


class ANTHROPIC(BASE_LLM_CACHE):
	"""
	A class to interact with Anthropic's language model while using a cache mechanism to optimize requests.
	Inherits from BASE_LLM_CACHE.
	"""
	def __init__(self, api_key, cache_collection=None, **kwargs):
		"""
		Initializes the ANTHROPIC class.

		Args:
			api_key (str): The API key to authenticate with Anthropic.
			cache_collection (MongoDB collection, optional): The cache collection to use for storing requests and responses.
			**kwargs: Additional keyword arguments for initializing the Anthropic client.
		"""
		self.llm_client = Anthropic(
			base_url=LLM.ANTHROPIC.BASE_URL,
			api_key=LLM.ANTHROPIC.API_KEY,
		)
		self.cost = dict()
		self.cache_collection = cache_collection
		if not self.cache_collection:
			self.cache_collection = MongoClient(
				MONGO.HOST,
				MONGO.PORT
				).llm.anthropic_cache

	@retry(RateLimitError, tries=3)
	def _call_model(self, query, use_cache):
		"""
		Calls the Anthropic model with the given query.

		Args:
			query (dict): The query to send to the Anthropic model.
			use_cache (bool): Whether to use cached responses if available.

		Returns:
			str: The response from the model, either from cache or a new request.
		"""
		response = NO_CACHE_YET
		if use_cache:
			response = self.check_cache(query)
		if response==NO_CACHE_YET:
			response = self.llm_client.messages.create(
				**query
			)
			usage = response.usage.model_dump()
			for k,v in usage.items():
				if type(v) is int:
					self.cost[k] = self.cost.get(k,0)+v
			response = [contentBlock.model_dump() for contentBlock in response.content]
			self.write_cache(query, response, usage)
		return response

class ANTHROPIC_SERVICE:
	"""
	A service class for managing ANTHROPIC LLM requests through a queue mechanism using MongoDB and RabbitMQ.
	"""
	collection = MongoClient(
		MONGO.HOST,
		MONGO.PORT
		).llm.anthropic
	queue_name = "llm-anthropic"

	logger = get_logger(
		__name__=__name__,
		__file__=__file__,
	)

	@staticmethod
	def trigger(
		parent_service: str,
		parent_job_id= None, # set this to be ObjectId if need callback
		use_cache=False,
		max_tokens=4096,
		**query
		) -> str:
		"""
		Creates and triggers a new job for an LLM request.

		Args:
			caller_service (str): The service initiating the request.
			use_cache (bool, optional): Whether to use cached responses if available.
			**query: The query to send to the LLM.

		Returns:
			str: The job ID of the triggered request.
		"""
		connection = pika.BlockingConnection(
			pika.ConnectionParameters(host='localhost'))
		channel = connection.channel()

		channel.queue_declare(
			queue=ANTHROPIC_SERVICE.queue_name,
			durable=True
		)
		query["max_tokens"] = max_tokens
		ANTHROPIC_SERVICE.logger.info("Writing job to Mongo")
		job_id = ANTHROPIC_SERVICE.collection.insert_one(
			dict(
				parent_service=parent_service,
				parent_job_id=parent_job_id,
				created_time = now(),
				use_cache=use_cache,
				query=query,
			)
		).inserted_id

		ANTHROPIC_SERVICE.logger.info("Pushing job to RabbitMQ")
		channel.basic_publish(
			exchange="",
			routing_key=ANTHROPIC_SERVICE.queue_name,
			body=str(job_id)
		)
		connection.close()

		ANTHROPIC_SERVICE.logger.info("Job pushed to RabbitMQ")
		return job_id

	@staticmethod
	def launch_worker():
		"""
		Launches a worker to process jobs from the RabbitMQ queue.
		The worker interacts with the LLM and stores the response back in MongoDB.
		"""
		try:
			connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
			channel = connection.channel()

			channel.queue_declare(
				queue=ANTHROPIC_SERVICE.queue_name,
				durable=True
			)
			llm_controller = ANTHROPIC(
				api_key=LLM.ANTHROPIC.API_KEY,
				base_url = LLM.ANTHROPIC.BASE_URL,
			)
			def callback(ch, method, properties, body):
				job_id = ObjectId(body.decode())
				ANTHROPIC_SERVICE.logger.info(f"Recieved LLM Query - {job_id}")
				job = ANTHROPIC_SERVICE.collection.find_one(dict(_id=job_id))
				query = job["query"]
				use_cache = job["use_cache"]
				
				parent_service = job["parent_service"]
				parent_job_id = job["parent_job_id"]

				ANTHROPIC_SERVICE.logger.debug(f"Recieved LLM Query - {query}")

				ret = llm_controller._call_model(
					query=query,
					use_cache=use_cache,
					)
				ANTHROPIC_SERVICE.collection.update_one(
					dict(_id=job_id),
					{"$set":dict(
						completion_time=now(),
						response=ret
					)}
				)
				ANTHROPIC_SERVICE.logger.debug(f"LLM Output - {ret}")
				if parent_job_id:
					parent_connection, parent_channel = get_channel(parent_service)
					parent_channel.basic_publish(
						exchange="",
						routing_key=parent_service,
						body=str(parent_job_id)
					)
					parent_connection.close()
				ch.basic_ack(delivery_tag = method.delivery_tag)
			channel.basic_consume(
				queue=ANTHROPIC_SERVICE.queue_name,
				on_message_callback=callback,
				auto_ack=False,
			)
			ANTHROPIC_SERVICE.logger.info('Worker Launched. To exit press CTRL+C')
			channel.start_consuming()
		except KeyboardInterrupt:
			ANTHROPIC_SERVICE.logger.warning('Shutting Off Worker')
			try:
				sys.exit(0)
			except SystemExit:
				os._exit(0)
	
	@staticmethod
	def get_response(job_id):
		"""
		Retrieves the response of a job with the given ID.

		Args:
			job_id (ObjectId): The ID of the job to retrieve the response for.

		Returns:
			str: The response of the job, or None if the job is not found or has not completed.
		"""
		record = ANTHROPIC_SERVICE.collection.find_one(dict(_id=job_id))
		if not record:
			ANTHROPIC_SERVICE.logger.error(f"Job With ID of {job_id} not found")
		elif "completion_time" not in record:
			ANTHROPIC_SERVICE.logger.error(f"Retrieving Response From Un-Finished Job With ID of {job_id}.")
		else:
			return record["response"]
		return None
	
	@staticmethod
	def get_response_sync(job_id, timeout=300):
		"""
		Retrieves the response of a job with the given ID synchronously.

		Args:
			job_id (ObjectId): The ID of the job to retrieve the response for.
			timeout (int, optional): The maximum time to wait for the job to complete.

		Returns:
			str: The response of the job, or None if the job is not found or has not completed within the timeout.
		"""
		start_time = time.time()
		while (timeout and (time.time() - start_time) < timeout):
			record = ANTHROPIC_SERVICE.collection.find_one(dict(_id=job_id))
			if not record:
				ANTHROPIC_SERVICE.logger.error(f"Job With ID of {job_id} not found")
				return None
			elif "completion_time" in record:
				if type(record["response"]) == list:
					return record["response"][0]["text"]
				else:
					return record["response"]
			time.sleep(1)  # Wait 1 second between checks
		
		ANTHROPIC_SERVICE.logger.error(f"Retrieving Response From Job {job_id} Timed Out After {timeout} Seconds")
		return None

	@staticmethod
	async def get_response_async(job_id, timeout=300):
		"""
		Retrieves the response of a job with the given ID asynchronously.

		Args:
			job_id (ObjectId): The ID of the job to retrieve the response for.
			timeout (int, optional): The maximum time to wait for the job to complete.

		Returns:
			str: The response of the job, or None if the job is not found or has not completed within the timeout.
		"""
		start_time = time.time()
		while (timeout and (time.time() - start_time) < timeout):
			record = ANTHROPIC_SERVICE.collection.find_one(dict(_id=job_id))
			if not record:
				ANTHROPIC_SERVICE.logger.error(f"Job With ID of {job_id} not found")
				return None
			elif "completion_time" in record:
				return record["response"]
			await asyncio.sleep(1)  # Wait 1 second between checks
		
		ANTHROPIC_SERVICE.logger.error(f"Retrieving Response From Job {job_id} Timed Out After {timeout} Seconds")
		return None
	
if __name__=="__main__":
	ANTHROPIC_SERVICE.logger.warning("STARTING LLM SERVICE")
	ANTHROPIC_SERVICE.launch_worker()