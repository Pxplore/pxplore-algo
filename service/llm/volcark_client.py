import logging
from volcenginesdkarkruntime import Ark
import httpx
from config import VOLCARK_API_KEY, MODEL_NAME_MAPPING

logger = logging.getLogger(__name__)

class VolcarkClient:

    client = Ark(
        api_key=VOLCARK_API_KEY,
        timeout=httpx.Timeout(timeout=1800)
    )

    @classmethod
    def call(cls, model, messages, **kwargs):
        """
        Generic method to call VolcArk API, handling both streaming and non-streaming cases
        
        Args:
            model: Model name to use
            messages: Array of message objects
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            For streaming=True: Returns the stream object
            For streaming=False: Returns the response content as a string
        """
        try:
            # Check if streaming is enabled
            stream = kwargs.get('stream', False)
            model = MODEL_NAME_MAPPING[model]
            
            if stream:
                # For streaming, return the stream object directly
                # The caller will iterate through it
                logger.info(f"STARTING STREAM REQUEST: model={model}, stream=True")
                
                return cls.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
            else:
                # For non-streaming, return the response content
                logger.info(f"Starting non-streaming request to VolcArk API with model {model}")
                
                response = cls.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content
                    logger.info(f"Received complete response: {len(content)} characters")
                    return content
                    
                logger.warning("Received empty response from VolcArk API")
                return ""
                
        except Exception as e:
            logger.error(f"Error in VolcArk API call: {str(e)}")
            raise e
