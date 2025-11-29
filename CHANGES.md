# Changes Log

## Feature: Conversation History Persistence and Editing

This update introduces the ability to persist LLM conversation history to a JSON file, allowing users to inspect and modify past conversations. The API client now dynamically loads and saves conversation states from this file, ensuring that manual edits are reflected in subsequent interactions.

### Detailed Changes:

**1. `llm_plugin.py`**

*   **`_get_or_create_client(window)` (Modified):**
    *   The `window_id` is now passed to `create_groq_sync_client` when initializing the LLM client. This links each conversation to a specific Sublime Text window, enabling unique conversation file management.
*   **`LlmDumpConversationCommand` (Added):**
    *   A new Sublime Text command (`llm_dump_conversation`) has been introduced.
    *   When executed, it calls the `dump_conversation_to_json` method on the active LLM client's executor.
    *   After successfully dumping the conversation, it automatically opens the generated JSON file in a new Sublime Text tab, providing immediate access for editing.
*   **`Default.sublime-commands` (Modified):**
    *   The `llm_dump_conversation` command has been added to the command palette.

**2. `LLMConnect/top.py`**

*   **Imports (Modified):**
    *   Added `os` for file system operations and `sublime` to access Sublime Text-specific paths (e.g., `sublime.cache_path()`).
*   **`CONVERSATION_DIR_NAME` (Added):**
    *   A new constant `CONVERSATION_DIR_NAME = "LLMConnect_conversations"` was defined to specify the subdirectory within Sublime Text's cache path where conversation JSON files will be stored.
*   **`APIExecutor.__init__` (Modified):**
    *   Now accepts an optional `window_id` parameter.
    *   Initializes `self.conversation_file_path` using `_get_conversation_file_path` if a `window_id` is provided.
    *   Automatically calls `_load_conversation_from_file()` upon initialization if a conversation file path is set and the file exists, ensuring previous conversations are loaded.
*   **`_get_conversation_file_path(self)` (Added):**
    *   A new private helper method that generates the absolute file path for a conversation JSON file based on the `window_id`.
    *   Creates the `LLMConnect_conversations` directory if it doesn't exist.
    *   Includes debug print statements for tracing.
*   **`_load_conversation_from_file(self)` (Added):**
    *   A new private method responsible for reading and parsing the JSON conversation file into the `self.messages` list.
    *   Handles cases where the file doesn't exist, is empty, or contains invalid JSON.
    *   Includes debug print statements for tracing.
*   **`_save_conversation_to_file(self)` (Added):**
    *   A new private method that serializes the current `self.messages` list into JSON format and writes it to the conversation file.
    *   Includes debug print statements for tracing.
*   **`dump_conversation_to_json(self)` (Added):**
    *   A new public method that explicitly triggers the saving of the current `self.messages` to the conversation JSON file.
    *   Ensures the `conversation_file_path` is set before attempting to save.
    *   Includes extensive debug print statements for tracing.
*   **`add_message(self, role, content)` (Modified):**
    *   After appending a new message to `self.messages`, this method now calls `_save_conversation_to_file()` to immediately persist the updated conversation to disk.
*   **`clear_messages(self)` (Modified):**
    *   In addition to clearing the `self.messages` list, this method now attempts to delete the associated conversation JSON file from disk, effectively resetting the conversation both in memory and on the file system.
*   **`prepare_request_data(self, prompt, stream)` (Modified):**
    *   Crucially, this method now calls `_load_conversation_from_file()` at the beginning of its execution. This ensures that before preparing any new request, the client's internal `messages` state is synchronized with the latest content from the conversation JSON file, allowing manual edits to take effect.
*   **`SyncAPIClient.__init__` and `AsyncAPIClient.__init__` (Modified):**
    *   Both constructors now accept an optional `window_id` parameter, which is then passed directly to the `APIExecutor`'s constructor.

**3. `LLMConnect/api_client_factory.py`**

*   **`APIClientFactory.create_sync_client` and `APIClientFactory.create_async_client` (Modified):**
    *   These static methods now accept an optional `window_id` parameter and correctly pass it down to the `SyncAPIClient` or `AsyncAPIClient` constructors.
*   **`_make_sync_client_func` and `_make_async_client_func` (Modified):**
    *   The dynamically generated client factory functions now also accept and pass through the `window_id` parameter, ensuring it's available when creating new clients.

These changes provide a robust mechanism for managing LLM conversation history, integrating file-based persistence and manual editing capabilities directly into the Sublime Text plugin workflow.
