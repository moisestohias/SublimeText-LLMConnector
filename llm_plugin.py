import sublime
import sublime_plugin
import threading
import os
import subprocess
from .LLMConnect import create_groq_sync_client

# Global dictionaries to store clients and conversation views per window
_window_clients = {}
_conversation_views = {}
delimiter = "==="

def _get_or_create_client(window):
    """Get or create LLM client for the given window."""
    window_id = window.id()
    
    if window_id not in _window_clients or _window_clients[window_id] is None:
        settings = sublime.load_settings('LLMPlugin.sublime-settings')
        client_config = settings.get('client_config', {})
        _window_clients[window_id] = create_groq_sync_client(**client_config)
    
    return _window_clients[window_id]


class LlmChatCommand(sublime_plugin.WindowCommand):
    def run(self):
        window = self.window
        view = self._get_or_create_conversation_view(window)
        window.focus_view(view)
        
        window.show_input_panel("Prompt:", "", self.on_done, None, None)

    def _get_or_create_conversation_view(self, window):
        window_id = window.id()
        view_id = _conversation_views.get(window_id)
        
        if view_id:
            view = next((v for v in window.views() if v.id() == view_id), None)
            if view:
                group, _ = window.get_view_index(view)
                window.focus_group(group)
                window.focus_view(view)
                return view
        
        if window.num_groups() < 2:
            window.set_layout({
                "cols": [0.0, 0.5, 1.0],
                "rows": [0.0, 1.0],
                "cells": [[0, 0, 1, 1], [1, 0, 2, 1]]
            })

        target_group = window.num_groups() - 1
        window.focus_group(target_group)

        view = window.new_file()
        view.set_name("LLM Conversation")
        view.set_scratch(True)
        view.set_syntax_file("Packages/Markdown/Markdown.sublime-syntax")
        
        _conversation_views[window_id] = view.id()
        
        return view

    def on_done(self, text):
        if not text:
            return

        window = self.window
        client = _get_or_create_client(window)
        view_id = _conversation_views.get(window.id())
        view = next((v for v in window.views() if v.id() == view_id), None)

        if view:
            prompt_text = '\n\n> User:\n{0}'.format(text)
            view.run_command('append', {'characters': prompt_text})
            
            thread = threading.Thread(target=self.do_request, args=(client, text, view))
            thread.start()

    def do_request(self, client, text, view):
        try:
            response = client.chat(text)
            sublime.set_timeout(lambda: self.on_response(response, view), 0)
        except Exception as e:
            error_message = 'LLM Plugin Error: {0}'.format(str(e))
            sublime.error_message(error_message)

    def on_response(self, response, view):
        response_text = '\n\n> Assistant:\n{0}'.format(response)
        view.run_command('append', {'characters': response_text})


class LlmQueryCommand(sublime_plugin.TextCommand):
    def run(self, edit):
        window = self.view.window()
        client = _get_or_create_client(window)
        
        # Collect all selections and their insert points before going async
        queries = []
        for sel in self.view.sel():
            if not sel.empty():
                selected_text = self.view.substr(sel)
                insert_point = sel.end()
                queries.append((selected_text, insert_point))
        
        if not queries:
            return
        
        # Show status message
        sublime.status_message("LLM: Processing query...")
        
        # Run API call in background thread
        thread = threading.Thread(
            target=self._do_request,
            args=(client, queries)
        )
        thread.start()
    
    def _do_request(self, client, queries):
        """Background thread: make API calls."""
        try:
            results = []
            for selected_text, insert_point in queries:
                response = client.chat(selected_text)
                results.append((insert_point, response))
            
            # Schedule UI update on main thread
            sublime.set_timeout(lambda: self._on_response(results), 0)
        
        except Exception as e:
            error_msg = 'LLM Plugin Error: {0}'.format(str(e))
            sublime.set_timeout(lambda: sublime.error_message(error_msg), 0)
    
    def _on_response(self, results):
        """Main thread: insert responses into the view."""
        # Sort by insert_point descending so insertions don't shift later positions
        results.sort(key=lambda x: x[0], reverse=True)
        
        for insert_point, response in results:
            response_text = '\n\n{0}\n{1}\n{0}\n'.format(delimiter, response)
            # Use run_command to get a fresh edit object
            self.view.run_command('llm_insert_text', {
                'point': insert_point,
                'text': response_text
            })
        
        self.view.sel().clear()
        sublime.status_message("LLM: Done")
    
    def is_enabled(self):
        return any(not sel.empty() for sel in self.view.sel())


class LlmInsertTextCommand(sublime_plugin.TextCommand):
    """Helper command to insert text at a specific position."""
    def run(self, edit, point, text):
        self.view.insert(edit, point, text)

class LlmResetConversationCommand(sublime_plugin.ApplicationCommand):
    def run(self):
        try:
            window = sublime.active_window()
            window_id = window.id()
            
            if window_id in _window_clients and _window_clients[window_id] is not None:
                _window_clients[window_id].clear_messages()
                sublime.status_message('LLM conversation history reset')

            view_id = _conversation_views.get(window_id)
            view = next((v for v in window.views() if v.id() == view_id), None)
            if view:
                view.run_command('select_all')
                view.run_command('right_delete')
                sublime.status_message('LLM conversation view cleared')
            
        except Exception as e:
            sublime.error_message('LLM Reset Error: {0}'.format(str(e)))

class LlmEventListener(sublime_plugin.EventListener):
    def on_close(self, view):
        view_id = view.id()
        window_id_to_del = None
        for wid, vid in _conversation_views.items():
            if vid == view_id:
                window_id_to_del = wid
                break
        
        if window_id_to_del:
            del _conversation_views[window_id_to_del]


def plugin_unloaded():
    """Clean up clients when plugin is unloaded."""
    global _window_clients, _conversation_views
    _window_clients.clear()
    _conversation_views.clear()
