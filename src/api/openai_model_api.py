import os

import openai
from openai._base_client import SyncHttpxClientWrapper

# import openai


_old_init = SyncHttpxClientWrapper.__init__


def new_init(self, *args, **kwargs):
    kwargs.pop("proxies", None)
    return _old_init(self, *args, **kwargs)


SyncHttpxClientWrapper.__init__ = new_init


from src.api.base_model_api import BaseModelAPI


class OpenAIModelAPI(BaseModelAPI):
    """
    A unified OpenAI API client that works with both legacy GPT models and the new reasoning models (o1/o3).
    """

    def __init__(self, api_key=None, model="gpt-4.1-mini"):
        super().__init__(api_key)
        self.api_key = api_key or self.get_api_key_from_env()
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        self.conversation_history = []

    def get_api_key_from_env(self):
        """Retrieve the OpenAI API key from environment variables."""
        return os.getenv("OPENAI_API_KEY")

    def get_response(self, prompt, max_tokens=1024, temperature=0.5):
        """
        Get a response from the OpenAI model.
        Uses `max_completion_tokens` (and omits temperature) for o1/o3 models,
        while legacy models use `max_tokens` and support the temperature parameter.
        """
        # Append the user prompt to the conversation history.
        self.conversation_history.append({"role": "user", "content": prompt})

        # Build the parameters for the API call.
        params = {
            "model": self.model,
            "messages": self.conversation_history,
        }
        # For new reasoning models (o1/o3), set the token limit using max_completion_tokens.
        # Also, these models have fixed settings for temperature (and others) so do not pass that parameter.
        if "o1" in self.model or "o3" in self.model:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = temperature

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        # Extract and store the assistant's reply.
        assistant_response = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )
        return assistant_response.strip()
