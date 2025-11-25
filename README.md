# LLM Connector Plugin Summary

## Overview

This Sublime Text plugin provides a seamless interface to interact with LLMs. It allows users to either query the LLM with selected text directly in the current file or engage in a persistent, back-and-forth conversation in a dedicated view.

## Operational Framework

The plugin's state is managed on a per-window basis using global dictionaries. This ensures that each Sublime Text window has its own independent LLM client, conversation history, and dedicated conversation view, preventing any overlap between different user sessions or projects.

### Core Components

-   `_window_clients`: A global dictionary that stores the initialized LLM API client for each window, mapping `window.id()` to the client instance. This lazy-loads clients, creating one only when first needed in a window.
-   `_conversation_views`: A global dictionary that maps a `window.id()` to the `view.id()` of its dedicated conversation tab. This allows commands to easily find and interact with the correct conversation view.
-   `_get_or_create_client(window)`: A helper function that centralizes client management. It checks if a client already exists for the given window; if not, it reads the configuration from `LLMPlugin.sublime-settings` and initializes a new one.

## Commands (Classes)

### `LlmChatCommand(sublime_plugin.WindowCommand)`

-   **Purpose**: To initiate and manage a continuous conversation with the LLM.
-   **Behavior**:
    1.  When triggered, it first checks if a conversation view already exists for the active window.
    2.  If not, it creates a two-column layout, generates a new, empty tab named "LLM Conversation" in the second column, and sets its syntax to Markdown for better readability. This view is marked as a "scratch" buffer, so it doesn't prompt the user to save changes.
    3.  It then displays an input panel at the bottom of the window, prompting the user for their message.
    4.  Upon submission, the user's prompt is appended to the conversation view.
    5.  The prompt is sent to the LLM API in a separate thread to prevent UI freezes.
    6.  Once the LLM's response is received, it is appended to the conversation view, completing the turn.

### `LlmQueryCommand(sublime_plugin.TextCommand)`

-   **Purpose**: To perform a one-off query using the currently selected text.
-   **Behavior**:
    1.  This command is only enabled when there is a non-empty text selection.
    2.  It takes the selected text and sends it as a prompt to the LLM.
    3.  The LLM's response is then inserted directly below the selection in the active buffer, separated by a configurable delimiter.

### `LlmResetConversationCommand(sublime_plugin.ApplicationCommand)`

-   **Purpose**: To clear the current conversation without resetting the underlying client connection.
-   **Behavior**:
    1.  It calls the `clear_messages()` method on the active window's LLM client to reset the conversation history.
    2.  It clears all the text content from the associated conversation view.
    3.  A status message confirms that the conversation has been reset.

### `LlmResetClientCommand(sublime_plugin.ApplicationCommand)`

-   **Purpose**: To completely reset the LLM client and close the conversation view.
-   **Behavior**:
    1.  It removes the LLM client instance for the active window from the `_window_clients` dictionary.
    2.  It closes the associated "LLM Conversation" tab.
    3.  This is useful for reloading configuration or resolving connection issues.

## Event Handling and Lifecycle

### `LlmEventListener(sublime_plugin.EventListener)`

-   **Purpose**: To automatically manage resources when the user manually closes the conversation view.
-   **Behavior**: The `on_close` event handler checks if the closed view was a conversation view. If so, it removes the corresponding entry from the `_conversation_views` dictionary to ensure a new one can be created next time.

### `plugin_unloaded()`

-   **Purpose**: To ensure a clean shutdown and resource cleanup.
-   **Behavior**: This function is called automatically by Sublime Text when the plugin is being unloaded or reloaded. It clears both the `_window_clients` and `_conversation_views` dictionaries, releasing all client instances and view references.

### `LlmTtsCommand(sublime_plugin.TextCommand)`

-   **Purpose**: To read the currently selected text aloud using a background script.
-   **Behavior**:
    1.  This command is enabled only when text is selected.
    2.  It takes the selected text and passes it as an argument to the `~/.script/tts` shell script.
    3.  The script is executed in the background, so it does not block the Sublime Text UI.