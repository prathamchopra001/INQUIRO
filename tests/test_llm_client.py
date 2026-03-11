"""
Tests for LLMClient.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock external imports if not installed
# We need to set these up BEFORE importing the module under test
mock_openai = MagicMock()
sys.modules["openai"] = mock_openai

mock_anthropic = MagicMock()
sys.modules["anthropic"] = mock_anthropic

mock_google = MagicMock()
sys.modules["google.generativeai"] = mock_google

from src.utils.llm_client import LLMClient, OpenAIClient, AnthropicClient


class TestLLMClient:
    """Tests for LLMClient factory and providers."""

    @patch("src.utils.llm_client.settings")
    def test_client_factory_openai(self, mock_settings):
        """Test creating OpenAI client."""
        mock_settings.llm.provider = "openai"
        mock_settings.llm.model = "gpt-4"
        mock_settings.llm.openai_api_key = "sk-test"

        with patch("src.utils.llm_client.OpenAIClient.is_available", return_value=True):
            client = LLMClient()
            assert isinstance(client._client, OpenAIClient)
            assert client.model == "gpt-4"

    @patch("src.utils.llm_client.settings")
    def test_client_factory_auto_detect(self, mock_settings):
        """Test auto-detection when configured provider fails."""
        mock_settings.llm.provider = "openai"
        mock_settings.llm.openai_api_key = None  # Not available

        # Mock Ollama as available
        with patch("src.utils.llm_client.OllamaClient.is_available", return_value=True):
             # Mock OpenAIClient.is_available to return False
            with patch("src.utils.llm_client.OpenAIClient.is_available", return_value=False):
                client = LLMClient()
                assert client._client.__class__.__name__ == "OllamaClient"

    def test_is_available_openai(self):
        """Test OpenAI is_available logic."""
        client = OpenAIClient(api_key="sk-test")

        # Mock check for package installed
        with patch.object(client, "_is_package_installed", return_value=True):
            assert client.is_available() is True

        with patch.object(client, "_is_package_installed", return_value=False):
            assert client.is_available() is False

        client_no_key = OpenAIClient(api_key=None)
        with patch.object(client_no_key, "_is_package_installed", return_value=True):
            assert client_no_key.is_available() is False

    def test_is_available_anthropic(self):
        """Test Anthropic is_available logic."""
        client = AnthropicClient(api_key="sk-ant-test")

        with patch.object(client, "_is_package_installed", return_value=True):
            assert client.is_available() is True

        with patch.object(client, "_is_package_installed", return_value=False):
            assert client.is_available() is False

    def test_openai_complete(self):
        """Test OpenAI completion."""
        client = OpenAIClient(api_key="sk-test", model="gpt-4")

        # Configure the global mock_openai
        # When 'from openai import OpenAI' happens, it gets mock_openai.OpenAI
        # So we configure mock_openai.OpenAI return value (the instance)
        mock_instance = MagicMock()
        mock_openai.OpenAI.return_value = mock_instance

        # Mock response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_instance.chat.completions.create.return_value = mock_response

        response = client.complete("Hello")

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage["total_tokens"] == 15
