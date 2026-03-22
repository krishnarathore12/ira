from abc import ABC
from typing import Dict, Optional, Union


class BaseLlmConfig(ABC):
    """
    Config for LLMs.
    """

    def __init__(
            self,
            model: Optional[Union[str, Dict]] = None,
            temperature: float = 0.1,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            max_tokens: int = 8192,
            top_p: float = 0.1,
            top_k: int = 1,
    ):
        """
        Initializes a configuration class instance for the LLM.

        :param model: Controls the OpenAI model used, defaults to None
        :type model: Optional[str], optional
        :param temperature:  Controls the randomness of the model's output.
        Higher values (closer to 1) make output more random, lower values make it more deterministic, defaults to 0
        :type temperature: float, optional
        :param api_key: OpenAI API key to be use, defaults to None
        :type api_key: Optional[str], optional
        :param max_tokens: Controls how many tokens are generated, defaults to 2000
        :type max_tokens: int, optional
        :param top_p: Controls the diversity of words. Higher values (closer to 1) make word selection more diverse,
        defaults to 1
        :type top_p: float, optional
        :param top_k: Controls the diversity of words. Higher values make word selection more diverse, defaults to 0
        :type top_k: int, optional
        :param base_url: Openai base URL to be used, defaults to "https://api.openai.com/v1"
        :type base_url: Optional[str], optional
        """

        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
