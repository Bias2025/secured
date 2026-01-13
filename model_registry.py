#!/usr/bin/env python3
"""
Model Registry for Garak Scanner
=================================

Maintains a registry of popular LLM models across different providers.
Allows users to add and manage custom models.

Author: Isi Idemudia
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelInfo:
    """Information about a specific model"""
    name: str
    provider: str
    display_name: str
    description: str
    context_window: Optional[int] = None
    cost_tier: Optional[str] = None  # free, low, medium, high
    release_date: Optional[str] = None
    is_custom: bool = False


class ModelRegistry:
    """Registry of LLM models with support for custom additions"""

    def __init__(self, custom_models_file: str = "custom_models.json"):
        self.custom_models_file = Path(custom_models_file)
        self.models = self._initialize_models()
        self._load_custom_models()

    def _initialize_models(self) -> Dict[str, List[ModelInfo]]:
        """Initialize registry with popular models"""
        return {
            "openai": [
                ModelInfo(
                    name="gpt-4o",
                    provider="openai",
                    display_name="GPT-4o",
                    description="Latest GPT-4 Optimized model - fastest and most capable",
                    context_window=128000,
                    cost_tier="high",
                    release_date="2024-05"
                ),
                ModelInfo(
                    name="gpt-4o-mini",
                    provider="openai",
                    display_name="GPT-4o Mini",
                    description="Affordable and intelligent small model",
                    context_window=128000,
                    cost_tier="low",
                    release_date="2024-07"
                ),
                ModelInfo(
                    name="gpt-4-turbo",
                    provider="openai",
                    display_name="GPT-4 Turbo",
                    description="High-capability model with vision",
                    context_window=128000,
                    cost_tier="high",
                    release_date="2024-04"
                ),
                ModelInfo(
                    name="gpt-4",
                    provider="openai",
                    display_name="GPT-4",
                    description="Original GPT-4 model",
                    context_window=8192,
                    cost_tier="high",
                    release_date="2023-03"
                ),
            ],

            "google": [
                ModelInfo(
                    name="gemini-2.0-flash-exp",
                    provider="google",
                    display_name="Gemini 2.0 Flash (Experimental)",
                    description="Newest experimental Gemini model",
                    context_window=1000000,
                    cost_tier="medium",
                    release_date="2024-12"
                ),
                ModelInfo(
                    name="gemini-1.5-pro",
                    provider="google",
                    display_name="Gemini 1.5 Pro",
                    description="Mid-size multimodal model with 1M token context",
                    context_window=1000000,
                    cost_tier="high",
                    release_date="2024-02"
                ),
                ModelInfo(
                    name="gemini-1.5-flash",
                    provider="google",
                    display_name="Gemini 1.5 Flash",
                    description="Fast and versatile multimodal model",
                    context_window=1000000,
                    cost_tier="low",
                    release_date="2024-05"
                ),
                ModelInfo(
                    name="gemini-pro",
                    provider="google",
                    display_name="Gemini Pro",
                    description="Best model for scaling across tasks",
                    context_window=32760,
                    cost_tier="medium",
                    release_date="2023-12"
                ),
            ],

            "anthropic": [
                ModelInfo(
                    name="claude-3-5-sonnet-20241022",
                    provider="anthropic",
                    display_name="Claude 3.5 Sonnet (Latest)",
                    description="Most intelligent Claude model - best for complex tasks",
                    context_window=200000,
                    cost_tier="high",
                    release_date="2024-10"
                ),
                ModelInfo(
                    name="claude-3-5-sonnet-20240620",
                    provider="anthropic",
                    display_name="Claude 3.5 Sonnet",
                    description="Intelligent model with balanced performance",
                    context_window=200000,
                    cost_tier="high",
                    release_date="2024-06"
                ),
                ModelInfo(
                    name="claude-3-opus-20240229",
                    provider="anthropic",
                    display_name="Claude 3 Opus",
                    description="Most powerful Claude model for complex tasks",
                    context_window=200000,
                    cost_tier="high",
                    release_date="2024-02"
                ),
                ModelInfo(
                    name="claude-3-sonnet-20240229",
                    provider="anthropic",
                    display_name="Claude 3 Sonnet",
                    description="Balanced intelligence and speed",
                    context_window=200000,
                    cost_tier="medium",
                    release_date="2024-02"
                ),
                ModelInfo(
                    name="claude-3-haiku-20240307",
                    provider="anthropic",
                    display_name="Claude 3 Haiku",
                    description="Fastest and most compact model",
                    context_window=200000,
                    cost_tier="low",
                    release_date="2024-03"
                ),
            ],

            "cohere": [
                ModelInfo(
                    name="command-r-plus",
                    provider="cohere",
                    display_name="Command R+",
                    description="Most powerful Cohere model for complex tasks",
                    context_window=128000,
                    cost_tier="high",
                    release_date="2024-04"
                ),
                ModelInfo(
                    name="command-r",
                    provider="cohere",
                    display_name="Command R",
                    description="Scalable model for RAG and tool use",
                    context_window=128000,
                    cost_tier="medium",
                    release_date="2024-03"
                ),
                ModelInfo(
                    name="command",
                    provider="cohere",
                    display_name="Command",
                    description="General-purpose model for various tasks",
                    context_window=4096,
                    cost_tier="medium",
                    release_date="2023-08"
                ),
                ModelInfo(
                    name="command-light",
                    provider="cohere",
                    display_name="Command Light",
                    description="Faster, more affordable model",
                    context_window=4096,
                    cost_tier="low",
                    release_date="2023-08"
                ),
            ],

            "huggingface": [
                ModelInfo(
                    name="meta-llama/Llama-3.2-3B-Instruct",
                    provider="huggingface",
                    display_name="Llama 3.2 3B Instruct",
                    description="Latest efficient Llama model",
                    context_window=8192,
                    cost_tier="free",
                    release_date="2024-09"
                ),
                ModelInfo(
                    name="meta-llama/Llama-3.1-8B-Instruct",
                    provider="huggingface",
                    display_name="Llama 3.1 8B Instruct",
                    description="Powerful open-source instruction-tuned model",
                    context_window=128000,
                    cost_tier="free",
                    release_date="2024-07"
                ),
                ModelInfo(
                    name="meta-llama/Llama-3.1-70B-Instruct",
                    provider="huggingface",
                    display_name="Llama 3.1 70B Instruct",
                    description="Large high-performance open model",
                    context_window=128000,
                    cost_tier="free",
                    release_date="2024-07"
                ),
                ModelInfo(
                    name="mistralai/Mistral-7B-Instruct-v0.3",
                    provider="huggingface",
                    display_name="Mistral 7B Instruct v0.3",
                    description="Efficient open model from Mistral AI",
                    context_window=32768,
                    cost_tier="free",
                    release_date="2024-05"
                ),
                ModelInfo(
                    name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    provider="huggingface",
                    display_name="Mixtral 8x7B Instruct",
                    description="Mixture of experts model with strong performance",
                    context_window=32768,
                    cost_tier="free",
                    release_date="2023-12"
                ),
                ModelInfo(
                    name="google/flan-t5-xxl",
                    provider="huggingface",
                    display_name="FLAN-T5 XXL",
                    description="Google's instruction-tuned T5 model",
                    context_window=2048,
                    cost_tier="free",
                    release_date="2022-10"
                ),
                ModelInfo(
                    name="tiiuae/falcon-40b-instruct",
                    provider="huggingface",
                    display_name="Falcon 40B Instruct",
                    description="Powerful open model from TII",
                    context_window=2048,
                    cost_tier="free",
                    release_date="2023-05"
                ),
            ],

            "ollama": [
                ModelInfo(
                    name="llama3.2",
                    provider="ollama",
                    display_name="Llama 3.2",
                    description="Latest Llama model (run locally)",
                    cost_tier="free",
                    release_date="2024-09"
                ),
                ModelInfo(
                    name="llama3.1",
                    provider="ollama",
                    display_name="Llama 3.1",
                    description="Advanced Llama model (run locally)",
                    cost_tier="free",
                    release_date="2024-07"
                ),
                ModelInfo(
                    name="mistral",
                    provider="ollama",
                    display_name="Mistral",
                    description="Efficient 7B model (run locally)",
                    cost_tier="free",
                    release_date="2023-09"
                ),
                ModelInfo(
                    name="mixtral",
                    provider="ollama",
                    display_name="Mixtral",
                    description="Mixture of experts 8x7B (run locally)",
                    cost_tier="free",
                    release_date="2023-12"
                ),
                ModelInfo(
                    name="phi3",
                    provider="ollama",
                    display_name="Phi-3",
                    description="Microsoft's small but powerful model",
                    cost_tier="free",
                    release_date="2024-04"
                ),
                ModelInfo(
                    name="gemma2",
                    provider="ollama",
                    display_name="Gemma 2",
                    description="Google's open model family",
                    cost_tier="free",
                    release_date="2024-06"
                ),
            ],

            "replicate": [
                ModelInfo(
                    name="meta/llama-3.1-405b-instruct",
                    provider="replicate",
                    display_name="Llama 3.1 405B",
                    description="Largest Llama model via Replicate",
                    context_window=128000,
                    cost_tier="high",
                    release_date="2024-07"
                ),
                ModelInfo(
                    name="meta/llama-3.1-70b-instruct",
                    provider="replicate",
                    display_name="Llama 3.1 70B",
                    description="High-performance Llama via Replicate",
                    context_window=128000,
                    cost_tier="medium",
                    release_date="2024-07"
                ),
            ],

            "test": [
                ModelInfo(
                    name="test",
                    provider="test",
                    display_name="Test Model",
                    description="Built-in test model for validation",
                    cost_tier="free",
                    release_date="2023-01"
                ),
            ],
        }

    def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """Get all models for a specific provider"""
        return self.models.get(provider, [])

    def get_all_providers(self) -> List[str]:
        """Get list of all supported providers"""
        return list(self.models.keys())

    def add_custom_model(self, model_info: ModelInfo) -> bool:
        """Add a custom model to the registry"""
        try:
            model_info.is_custom = True

            if model_info.provider not in self.models:
                self.models[model_info.provider] = []

            # Check if model already exists
            existing = [m for m in self.models[model_info.provider] if m.name == model_info.name]
            if existing:
                # Update existing model
                idx = self.models[model_info.provider].index(existing[0])
                self.models[model_info.provider][idx] = model_info
            else:
                # Add new model
                self.models[model_info.provider].append(model_info)

            self._save_custom_models()
            return True
        except Exception as e:
            print(f"Error adding custom model: {e}")
            return False

    def remove_custom_model(self, provider: str, model_name: str) -> bool:
        """Remove a custom model from the registry"""
        try:
            if provider in self.models:
                self.models[provider] = [
                    m for m in self.models[provider]
                    if not (m.name == model_name and m.is_custom)
                ]
                self._save_custom_models()
                return True
            return False
        except Exception as e:
            print(f"Error removing custom model: {e}")
            return False

    def get_custom_models(self) -> List[ModelInfo]:
        """Get all custom models"""
        custom = []
        for models in self.models.values():
            custom.extend([m for m in models if m.is_custom])
        return custom

    def _save_custom_models(self):
        """Save custom models to file"""
        try:
            custom_models = self.get_custom_models()
            data = [asdict(m) for m in custom_models]

            with open(self.custom_models_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving custom models: {e}")

    def _load_custom_models(self):
        """Load custom models from file"""
        try:
            if self.custom_models_file.exists():
                with open(self.custom_models_file, 'r') as f:
                    data = json.load(f)

                for model_data in data:
                    model_info = ModelInfo(**model_data)
                    if model_info.provider not in self.models:
                        self.models[model_info.provider] = []

                    # Add if not already present
                    if not any(m.name == model_info.name for m in self.models[model_info.provider]):
                        self.models[model_info.provider].append(model_info)
        except Exception as e:
            print(f"Error loading custom models: {e}")

    def search_models(self, query: str) -> List[ModelInfo]:
        """Search models by name or description"""
        query = query.lower()
        results = []

        for models in self.models.values():
            for model in models:
                if (query in model.name.lower() or
                    query in model.display_name.lower() or
                    query in model.description.lower()):
                    results.append(model)

        return results

    def get_model_info(self, provider: str, model_name: str) -> Optional[ModelInfo]:
        """Get detailed info for a specific model"""
        if provider in self.models:
            for model in self.models[provider]:
                if model.name == model_name:
                    return model
        return None
