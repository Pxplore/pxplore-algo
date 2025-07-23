import argparse
import os

# ANSI color codes
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_RESET = "\033[0m"

def get_env_or_default(env_var_name, fallback):
    """
    Returns the value of the environment variable `env_var_name` if it exists,
    otherwise returns `fallback`.
    """
    return os.environ.get(env_var_name, fallback)

try:
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default=get_env_or_default("LOG", "INFO"),
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--zhipu_api_key",
        default=get_env_or_default("ZHIPU_API_KEY", None),
        help="Set ZHIPU service api_key",
    )
    parser.add_argument(
        "--openai_api_key",
        default=get_env_or_default("OPENAI_API_KEY", None),
        help="Set OPENAI service api_key",
    )
    parser.add_argument(
        "--openai_baseurl",
        default=get_env_or_default("OPENAI_BASEURL", None),
        help="Set OPENAI service baseurl",
    )
    parser.add_argument(
        "--volcark_api_key",
        default=get_env_or_default("VOLCARK_API_KEY", None),
        help="Set VOLCARK service api_key",
    )
    args = parser.parse_args()
except SystemExit:
    # If argparse fails (likely due to being called in a context where
    # command-line arguments aren't valid or available), use environment
    # variables with fallbacks directly.
    args = argparse.Namespace(
        log=get_env_or_default("LOG", "INFO"),
        zhipu_api_key=get_env_or_default("ZHIPU_API_KEY", None),
        openai_api_key=get_env_or_default("OPENAI_API_KEY", None),
        openai_baseurl=get_env_or_default("OPENAI_BASEURL", None),
        volcark_api_key=get_env_or_default("VOLCARK_API_KEY", None),
    )


def masked_value_color(val, is_api=False):
    """
    Return 'None' in red if val is None, empty, or equals 'xxx'.
    Otherwise, return 'set' in green.
    """
    if val is None or val.strip() == "" or val == "xxx":
        return f"{COLOR_RED}NONE{COLOR_RESET}"
    elif is_api:
        return f"{COLOR_GREEN}SET{COLOR_RESET}"
    else:
        return f"{COLOR_GREEN}{val}{COLOR_RESET}"


# Print masked values
print("Log level:         ", masked_value_color(args.log))
print("ZHIPU API Key:     ", masked_value_color(args.zhipu_api_key, is_api=True))
print("OpenAI API Key:    ", masked_value_color(args.openai_api_key, is_api=True))
print("OpenAI Base URL:   ", masked_value_color(args.openai_baseurl))
print("Volcark API Key:   ", masked_value_color(args.volcark_api_key, is_api=True))


class MONGO:
	HOST = "localhost"
	PORT = 27117

class LLM:
	class ZHIPU:
		API_KEY = args.zhipu_api_key

	class OPENAI:
		API_KEY = args.openai_api_key
		BASE_URL = args.openai_baseurl
		
	class VOLCARK:
		API_KEY = args.volcark_api_key
		MODEL_NAME_MAPPING = {
			"deepseek-r1": "ep-20250222122655-tfzjr"
		}

class EMBED:
	MODEL = "baai/bge-large-zh-v1.5"
	PROTOCOL = "http"
	HOST = "0.0.0.0" #  the host of the embed model
	PORT = 6633
      
class QDRANT:
	COLLECTION = "pxplore"
	HOST = "0.0.0.0" #  the host of the qdrant
	PORT = 6634
	GRPC_PORT = 6635

class LOG:
	LEVEL=args.log