from typing import Optional, List, Mapping, Any
from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
import requests

class MyLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "Qwen2-7B-Instruct"
    dummy_response: str = "My response"
    model_url: str = "http://127.0.0.1:6006/predict"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        try:
            # Prepare the payload for your model request
            payload = {
                "name": "Qwen2-7B-Instruct",
                "input": {
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.1,
                    "top_p": 0.1
                },
                "request_metadata": {
                    "user_id": "user123",
                    "timestamp": "2024-09-05T10:00:00Z"
                }
            }

            # Send the request to the deployed model
            response = requests.post(self.model_url, json=payload)
            # Extract the result from the response
            if response.status_code == 200:
                result = response.json()
                # Assuming the response contains the completion text
                return CompletionResponse(text=result.get('data')[0].get("text", ""))
            else:
                return CompletionResponse(text=f"Error: {response.status_code}")

        except Exception as e:
            return CompletionResponse(text=f"Exception occurred: {str(e)}")

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)