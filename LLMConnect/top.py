"""
HTTP LLM-Providers API Client using only Python standard library.
This was the original version. I've added send & post to both SyncAPIClient & AsyncAPIClient
"""

import json
import asyncio
import os
import sublime
from typing import Dict, List, Optional, Any, AsyncIterator, Union, Iterator

from .base import SyncHTTPClient, AsyncHTTPClient, ConnectionPool, RetryConfig
from .middlewares import AuthenticationMiddleware, UserAgentMiddleware, LoggingMiddleware, HTTPResponse, BaseMiddleware

from .utils import validate_messages_format

CONVERSATION_DIR_NAME = "LLMConnect_conversations"
user_agent: str = "APIClient/1.0.0"


# Core API Logic
class APIExecutor:
    """
    Core API logic shared between sync and async clients.
    It's designed specifically for an OpenAI-compatible "chat/completions" endpoint.
    Which is a Tightly Coupled design, A better approach would be, 
    Decouple the generic API logic from the specific "chat" logic. 
    `APIExecutor` could be refactored into a `BaseAPIExecutor`, with a `ChatAPIExecutor` 
    subclass that manages message history and chat-specific data formatting.
    """

    def __init__(self, api_key: str, base_url: str, model: str, 
                 endpoint: str = "chat/completions",
                 temperature: float = 0.7,
                 max_completion_tokens: int = 100,
                 timeout: float = 30.0,
                 window_id: Optional[int] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.endpoint = endpoint.rstrip('/').lstrip('/')
        self.timeout = timeout

        # LLM-specific parameters
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.messages = []
        self.window_id = window_id
        self.conversation_file_path = self._get_conversation_file_path() if window_id else None
        
        if self.conversation_file_path:
            self._load_conversation_from_file()

    def _get_conversation_file_path(self) -> Optional[str]:
        """Generates the conversation file path based on the window ID."""
        if not self.window_id:
            print("[LLMConnect] No window ID provided, cannot generate conversation file path.")
            return None
        cache_dir = os.path.join(sublime.cache_path(), CONVERSATION_DIR_NAME)
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, f"conversation_{self.window_id}.json")
        print(f"[LLMConnect] Generated conversation file path: {file_path}")
        return file_path

    def _load_conversation_from_file(self):
        """Loads conversation messages from the JSON file if it exists."""
        if self.conversation_file_path and os.path.exists(self.conversation_file_path):
            try:
                with open(self.conversation_file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    if file_content: # Only load if file is not empty
                        loaded_messages = json.loads(file_content)
                        if isinstance(loaded_messages, list) and all(isinstance(m, dict) for m in loaded_messages):
                            self.messages = loaded_messages
                        else:
                            print(f"[LLMConnect] Warning: Invalid conversation format in {self.conversation_file_path}")
                            self.messages = []
                    else:
                        self.messages = []
            except (json.JSONDecodeError, IOError) as e:
                print(f"[LLMConnect] Error loading conversation from {self.conversation_file_path}: {e}")
                self.messages = []
        else:
            self.messages = []

    def _save_conversation_to_file(self):
        """Saves the current conversation messages to the JSON file."""
        if self.conversation_file_path:
            try:
                with open(self.conversation_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.messages, f, indent=4)
                print(f"[LLMConnect] Conversation saved to {self.conversation_file_path}")
            except IOError as e:
                print(f"[LLMConnect] Error saving conversation to {self.conversation_file_path}: {e}")
        else:
            print("[LLMConnect] No conversation file path, skipping save.")

    def dump_conversation_to_json(self):
        """Public method to force saving the conversation to JSON."""
        print(f"[LLMConnect] Attempting to dump conversation for window ID: {self.window_id}")
        if not self.conversation_file_path:
            self.conversation_file_path = self._get_conversation_file_path()
            if not self.conversation_file_path:
                print("[LLMConnect] Error: Cannot dump conversation without a valid conversation file path.")
                raise RuntimeError("Cannot dump conversation without a window ID.")
        self._save_conversation_to_file()
        print(f"[LLMConnect] Conversation dump initiated to: {self.conversation_file_path}")

    def set_parameters(self, model: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_completion_tokens: Optional[int] = None):
        """Update LLM parameters."""
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if max_completion_tokens is not None:
            self.max_completion_tokens = max_completion_tokens

    def clear_messages(self):
        """Clear the message history and delete the conversation file if it exists."""
        self.messages = []
        if self.conversation_file_path and os.path.exists(self.conversation_file_path):
            try:
                os.remove(self.conversation_file_path)
                print(f"[LLMConnect] Conversation file deleted: {self.conversation_file_path}")
            except IOError as e:
                print(f"[LLMConnect] Error deleting conversation file {self.conversation_file_path}: {e}")

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history and save to file if tracking is enabled."""
        self.messages.append({"role": role, "content": content})
        self._save_conversation_to_file()

    def prepare_request_data(self, prompt: Union[str, List[Dict[str, str]]], stream: bool = False) -> Dict[str, Any]:
        """Prepare the request data for the API call."""
        # Always reload messages from file before preparing request to reflect external edits
        if self.conversation_file_path:
            self._load_conversation_from_file()

        if isinstance(prompt, str):
            # Add user message to history
            self.add_message("user", prompt)
        elif isinstance(prompt, list):
            validate_messages_format(prompt)
            self.messages = prompt
        else:
            raise TypeError(f"Prompt must be either a string or a list of message dictionaries, got {type(prompt)!r}")


        data = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens
        }

        if stream:
            data["stream"] = True

        return data

    def get_request_config(self, data: Dict[str, Any],
                          headers: Optional[Dict[str, str]] = None) -> tuple:
        """Get the request configuration (URL, headers, body)."""
        url = f"{self.base_url}/{self.endpoint}"
        body = json.dumps(data).encode('utf-8')

        request_headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.api_key}"
        }

        if headers:
            request_headers.update(headers)

        return url, request_headers, body

    def process_non_streaming_response(self, response: HTTPResponse) -> str:
        """Process a non-streaming chat response."""
        result = json.loads(response.body.decode('utf-8'))
        assistant_message = result["choices"][0]["message"]["content"]

        # Add assistant's response to history
        self.add_message("assistant", assistant_message)

        return assistant_message

    def parse_streaming_chunk(self, chunk: bytes) -> Optional[str]:
        """Parse a streaming chunk and extract content."""
        chunk_str = chunk.decode('utf-8', errors='ignore')
        
        # Handle Server-Sent Events format
        lines = chunk_str.strip().split('\n')
        content_parts = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("data: "):
                chunk_data = line[6:].strip()
                
                if chunk_data == "[DONE]":
                    return None  # End of stream
                
                if chunk_data:  # Skip empty data lines
                    try:
                        parsed = json.loads(chunk_data)
                        if "choices" in parsed and len(parsed["choices"]) > 0:
                            delta = parsed["choices"][0].get("delta", {})
                            if "content" in delta:
                                content_parts.append(delta["content"])
                    except json.JSONDecodeError:
                        # Log but don't fail on parse errors
                        logger.debug(f"Failed to parse SSE data: {chunk_data}")
                        continue
        
        # Return concatenated content from all events in this chunk
        return "".join(content_parts) if content_parts else ""

# Synchronous API Client
class SyncAPIClient:
    """Synchronous LLM API client."""

    def __init__(self, api_key: str, base_url: str, model: str,
                 endpoint: str = "chat/completions",
                 temperature: float = 0.7,
                 max_completion_tokens: int = 100,
                 timeout: float = 30.0,
                 http_client: Optional[SyncHTTPClient] = None,
                 middleware: Optional[List[BaseMiddleware]] = None,
                 connection_pool: Optional[ConnectionPool] = None,
                 retry_config: Optional[RetryConfig] = None,
                 window_id: Optional[int] = None):

        self._executor = APIExecutor(
            api_key, base_url, model, endpoint, temperature, max_completion_tokens, timeout, window_id
        )

        # Use provided client or create a new one
        if http_client:
            self._http_client = http_client
            self._owns_client = False
        else:
            # Create default middleware if none provided
            if middleware is None:
                middleware = [
                    AuthenticationMiddleware(api_key),
                    UserAgentMiddleware(user_agent),
                    LoggingMiddleware()
                ]

            self._http_client = SyncHTTPClient(connection_pool, retry_config, middleware)
            self._owns_client = True

    @property
    def model(self) -> str:
        return self._executor.model

    @property
    def temperature(self) -> float:
        return self._executor.temperature

    @property
    def max_completion_tokens(self) -> int:
        return self._executor.max_completion_tokens

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._executor.messages

    def set_parameters(self, model: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_completion_tokens: Optional[int] = None):
        """Update LLM parameters."""
        self._executor.set_parameters(model, temperature, max_completion_tokens)

    def clear_messages(self):
        """Clear the message history."""
        self._executor.clear_messages()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self._executor.add_message(role, content)

    def post(self, url: str, headers: Optional[Dict[str, str]] = None,
             body: Optional[bytes] = None, timeout: Optional[float] = None) -> HTTPResponse:
        """Send a synchronous POST request."""
        return self._http_client.request('POST', url, headers=headers, body=body, timeout=timeout)

    def send(self, messages_for_request: List[dict]) -> List[dict]:
        data = self._executor.prepare_request_data(messages_for_request, stream=False)
        url, headers, body = self._executor.get_request_config(data)
        response = self.post(url, headers=headers, body=body, timeout=self._executor.timeout)
        assistant_message = self._executor.process_non_streaming_response(response)
        return assistant_message

    def chat(self, prompt: str, stream: bool = False) -> Union[str, Iterator[str]]:
        """Send a chat message and get a response."""
        data = self._executor.prepare_request_data(prompt, stream)

        if stream:
            return self._stream_chat(data)
        else:
            url, headers, body = self._executor.get_request_config(data)
            response = self._http_client.request('POST', url, headers=headers,
                                               body=body, timeout=self._executor.timeout)
            return self._executor.process_non_streaming_response(response)

    def _stream_chat(self, data: Dict[str, Any]) -> Iterator[str]:
        """Handle streaming chat responses."""
        url, headers, body = self._executor.get_request_config(data)
        headers['Accept'] = 'text/event-stream'

        full_response = []

        for chunk in self._http_client.stream_request('POST', url, headers=headers,
                                                    body=body, timeout=self._executor.timeout):
            content = self._executor.parse_streaming_chunk(chunk)

            if content is None:  # End of stream
                break
            elif content:  # Non-empty content
                full_response.append(content)
                yield content

        # Add complete response to history
        if full_response:
            self._executor.add_message("assistant", "".join(full_response))

    def close(self):
        """Close the client and cleanup resources."""
        if self._owns_client:
            self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Asynchronous API Client
class AsyncAPIClient:
    """Asynchronous LLM API client."""

    def __init__(self, api_key: str, base_url: str, model: str,
                endpoint: str = "chat/completions",
                temperature: float = 0.7,
                max_completion_tokens: int = 100,
                timeout: float = 30.0,
                http_client: Optional[AsyncHTTPClient] = None,
                middleware: Optional[List[BaseMiddleware]] = None,
                connection_pool: Optional[ConnectionPool] = None,
                retry_config: Optional[RetryConfig] = None,
                window_id: Optional[int] = None):

        self._executor = APIExecutor(
            api_key, base_url, model, endpoint, temperature, max_completion_tokens, timeout, window_id
        )

        # Use provided client or create a new one
        if http_client:
            self._http_client = http_client
            self._owns_client = False
        else:
            # Create default middleware if none provided
            if middleware is None:
                middleware = [
                    AuthenticationMiddleware(api_key),
                    UserAgentMiddleware(user_agent),
                    LoggingMiddleware()
                ]

            self._http_client = AsyncHTTPClient(connection_pool, retry_config, middleware)
            self._owns_client = True

    @property
    def model(self) -> str:
        return self._executor.model

    @property
    def temperature(self) -> float:
        return self._executor.temperature

    @property
    def max_completion_tokens(self) -> int:
        return self._executor.max_completion_tokens

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._executor.messages

    def set_parameters(self, model: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_completion_tokens: Optional[int] = None):
        """Update LLM parameters."""
        self._executor.set_parameters(model, temperature, max_completion_tokens)

    def clear_messages(self):
        """Clear the message history."""
        self._executor.clear_messages()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self._executor.add_message(role, content)

    async def post(self, url: str, headers: Optional[Dict[str, str]] = None,
             body: Optional[bytes] = None, timeout: Optional[float] = None) -> HTTPResponse:
        """Send a asynchronous POST request."""
        return await self._http_client.request('POST', url, headers=headers, body=body, timeout=timeout)

    async def send(self, messages_for_request: List[dict]) -> List[dict]:
        data = self._executor.prepare_request_data(messages_for_request, stream=False)
        url, headers, body = self._executor.get_request_config(data)
        response = await self.post(url, headers=headers, body=body, timeout=self._executor.timeout)
        assistant_message = self._executor.process_non_streaming_response(response)
        return assistant_message

    async def chat(self, prompt: str, stream: bool = False) -> Union[str, AsyncIterator[str]]:
        """Send a chat message and get a response."""
        data = self._executor.prepare_request_data(prompt, stream)

        if stream:
            # Don't await here - return the async generator directly
            return self._stream_chat(data)
        else:
            url, headers, body = self._executor.get_request_config(data)
            response = await self._http_client.request('POST', url, headers=headers,
                                                     body=body, timeout=self._executor.timeout)
            return self._executor.process_non_streaming_response(response)

    async def _stream_chat(self, data: Dict[str, Any]) -> AsyncIterator[str]:
        """Handle streaming chat responses."""
        url, headers, body = self._executor.get_request_config(data)
        headers['Accept'] = 'text/event-stream'

        full_response = []

        async for chunk in self._http_client.stream_request('POST', url, headers=headers,
                                                          body=body, timeout=self._executor.timeout):
            content = self._executor.parse_streaming_chunk(chunk)

            if content is None:  # End of stream
                break
            elif content:  # Non-empty content
                full_response.append(content)
                yield content

        # Add complete response to history
        if full_response:
            self._executor.add_message("assistant", "".join(full_response))

    async def close(self):
        """Close the client and cleanup resources."""
        if self._owns_client:
            await self._http_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
