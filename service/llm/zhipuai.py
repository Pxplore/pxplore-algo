import pika
import sys
import os
import json
import asyncio

import aio_pika

from pymongo import MongoClient
from bson import ObjectId
from retry import retry
from zhipuai import ZhipuAI, APIConnectionError, APITimeoutError

from config import LLM, MONGO
from service.llm.base import BASE_LLM_CACHE, NO_CACHE_YET
from utils import now, get_logger


logger = get_logger(
	__name__=__name__,
	__file__=__file__,
	)


class ZHIPUAI(BASE_LLM_CACHE):
	def __init__(self, api_key, cache_collection=None, **kwargs):
		"""
		Initializes the ZHIPUAI class.

		Args:
			api_key (str): The API key to authenticate with ZhipuAI.
			cache_collection (MongoDB collection, optional): The cache collection to use for storing requests and responses.
			**kwargs: Additional keyword arguments for initializing the ZhipuAI client.
		"""
		self.llm_client = ZhipuAI(api_key=api_key, **kwargs)
		self.cost = dict()
		if cache_collection is None:
			cache_collection = MongoClient(
				MONGO.HOST,
				MONGO.PORT
			).llm.zhipuai_cache
		self.cache_collection = cache_collection

	@retry((APIConnectionError, APITimeoutError), tries=3)
	def call_model(self, query: dict, use_cache: bool, stream=None):
		"""
		Calls the ZhipuAI model with the given query.

		Args:
			query (dict): The query to send to the OpenAI model.
			use_cache (bool): Whether to use cached responses if available.

		Returns:
			str: The response from the model, either from cache or a new request.
		"""
		if use_cache:
			response = self.check_cache(query)
			if response is not NO_CACHE_YET:
				return response
		if "model" not in query:
			query["model"] = 'glm-4'
		if query.get("stream", False) and stream:
			full_response = ""
			for response in self.llm_client.chat.completions.create(
				**query
				):
				logger.debug(f"STREAMING: {response}")
				stream_text = response.choices[0].delta.content
				full_response = full_response + stream_text
				stream(stream_text)
		else:
			response = self.llm_client.chat.completions.create(
				**query
			)
			full_response = response.choices[0].message.content
		usage = dict(response.usage)
		for k, v in usage.items():
			if type(v) is int:
				cost_value = self.cost.get(k, 0)
				self.cost[k] = cost_value + v
			elif type(v) is dict:
				for inner_k, inner_v in v.items():
					inner_name = f'{k}.{inner_k}'
					cost_value = self.cost.get(inner_name, 0)
					self.cost[inner_name] = cost_value + inner_v

		response = full_response
		self.write_cache(query, response, usage)
		return response
	



class ZHIPUAI_SERVICE:
	"""
	A service class for managing ZhipuAI LLM requests through a queue mechanism using MongoDB and RabbitMQ.
	"""
	collection = MongoClient(
		MONGO.HOST,
		MONGO.PORT
		).llm.zhipu
	queue_name = 'llm-zhipu'

	@classmethod
	def trigger(cls, query: dict, caller_service: str, use_cache=False, stream_info=None) -> str:
		"""
		Creates and triggers a new job for an LLM request.

		Args:
			query: The query to send to the LLM.
			caller_service (str): The service initiating the request.
			use_cache (bool): Whether to use cached responses if available.

		Returns:
			str: The job ID of the triggered request.
		"""
		connection = pika.BlockingConnection(
			pika.ConnectionParameters(host='localhost'))
		channel = connection.channel()

		channel.queue_declare(
			queue=cls.queue_name,
			durable=True
		)
		logger.info('Writing job to Mongo')
		job_id = cls.collection.insert_one(
			dict(
				caller_service=caller_service,
				created_time = now(),
				use_cache=use_cache,
				query=query,
				stream_info=stream_info,
			)
		).inserted_id

		logger.info('Pushing job to RabbitMQ')
		channel.basic_publish(
			exchange='',
			routing_key=cls.queue_name,
			body=str(job_id)
		)
		connection.close()
		
		logger.info('Job pushed to RabbitMQ')
		return job_id

	@classmethod
	def launch_worker(cls):
		"""
		Launches a worker to process jobs from the RabbitMQ queue.
		The worker interacts with the LLM and stores the response back in MongoDB.
		"""
		try:
			connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
			channel = connection.channel()

			channel.queue_declare(
				queue=cls.queue_name,
				durable=True
			)
			llm_controller = ZHIPUAI(
				api_key=LLM.ZHIPUAI.API_KEY
			)

			def callback(ch, method, properties, body):
				job_id = ObjectId(body.decode())
				job = cls.collection.find_one(dict(_id=job_id))
				query, use_cache, stream_info = job['query'], job['use_cache'], job['stream_info']
				logger.debug(f'Received LLM Query - {query}')
				if stream_info:
					ret = llm_controller.call_model(
						query=query,
						use_cache=use_cache,
						stream=lambda x: ZHIPUAI_SERVICE.publish_from_llm_stream(
								content=x,
								**stream_info
							)
					)
				else:
					ret = llm_controller.call_model(
						query=query,
						use_cache=use_cache,
					)
				cls.collection.update_one(
					dict(_id=job_id),
					{'$set': dict(
						completion_time=now(),
						response=ret
					)}
				)
				logger.debug(f'LLM Output - {ret}')
				# TODO: parent job
				ch.basic_ack(delivery_tag = method.delivery_tag)
				
			channel.basic_consume(
				queue=cls.queue_name,
				on_message_callback=callback,
				auto_ack=False
			)
			logger.info('Worker Launched. To exit press CTRL+C')
			channel.start_consuming()

		except KeyboardInterrupt:
			logger.info('Shutting Off Worker')
			try:
				sys.exit(0)
			except SystemExit:
				os._exit(0)

	@classmethod
	def check_llm_job_done(cls, job_id):
		"""
		Checks if a job has been completed.

		Args:
			job_id (str): The ID of the job to check.

		Returns:
			bool: Whether the job has been completed.
		"""
		job = cls.collection.find_one(dict(_id=job_id))
		if job is None:
			raise Exception
		return ('response' in job)


	@classmethod
	def get_llm_job_response(cls, job_id):
		"""
		Gets the response of a completed job.

		Args:
			job_id (str): The ID of the job to check.

		Returns:
			str: The response of the job.
		"""
		job = cls.collection.find_one(dict(_id=job_id))
		if job is None:
			raise Exception
		logger.info(f"printing get llm job response {job.get('response', None)}")
		return job.get('response', None)

	@staticmethod
	def publish_from_llm_stream(
		queue_name: str,
		routing_key: str,
		content: dict,
	):
		async def _publish():
			connection = await aio_pika.connect_robust("amqp://localhost")
			try:
				channel = await connection.channel()
				exchange = await channel.declare_exchange(
					queue_name, aio_pika.ExchangeType.TOPIC
				)

				message = aio_pika.Message(
					body=json.dumps({
						"type": "revise_last_msg",
						"content": content,
					}).encode(),
					headers={"final": False},
				)

				await exchange.publish(
					message,
					routing_key=routing_key,
				)
			finally:
				await connection.close()

		# run the coroutine from sync context
		asyncio.run(_publish())


if __name__ == '__main__':
	logger.warning('STARTING LLM SERVICE')
	ZHIPUAI_SERVICE.launch_worker()
