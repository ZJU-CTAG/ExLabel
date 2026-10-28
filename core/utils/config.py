import os
import yaml
import logging
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for ExLabel.
    
    Loads model configurations from config.yaml and provides
    easy access to LLM and Embedder settings with independent selection.
    """
    
    def __init__(self, config_path: str = "config.yaml", llm_name: str = None, embedder_name: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the config.yaml file
            llm_name: Name of the LLM to use (e.g., 'deepseek-chat', 'gpt-4')
            embedder_name: Name of the embedder to use (e.g., 'Qwen3-Embedding-8B')
        """
        self.config_path = config_path
        self.config_data = self._load_config()
        
        if not llm_name:
            available = ", ".join(self.config_data.get("llms", {}).keys())
            raise ValueError(
                f"LLM name must be provided. Available LLMs: {available}"
            )
        
        if not embedder_name:
            available = ", ".join(self.config_data.get("embedders", {}).keys())
            raise ValueError(
                f"Embedder name must be provided. Available embedders: {available}"
            )
        
        self.llm_name = llm_name
        self.embedder_name = embedder_name
        
        if self.llm_name not in self.config_data.get("llms", {}):
            available = ", ".join(self.config_data.get("llms", {}).keys())
            raise ValueError(
                f"LLM '{self.llm_name}' not found in {config_path}. "
                f"Available LLMs: {available}"
            )
        
        if self.embedder_name not in self.config_data.get("embedders", {}):
            available = ", ".join(self.config_data.get("embedders", {}).keys())
            raise ValueError(
                f"Embedder '{self.embedder_name}' not found in {config_path}. "
                f"Available embedders: {available}"
            )
        
        self.llm_config = self.config_data["llms"][self.llm_name]
        self.embedder_config = self.config_data["embedders"][self.embedder_name]
        
        logging.info(f"Using LLM: {self.llm_name}")
        logging.info(f"Using Embedder: {self.embedder_name}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Parsed configuration dictionary
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}. "
                f"Please create config.yaml in the project root directory."
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse {self.config_path}: {e}")
    
    @property
    def llm_model_name(self) -> str:
        """Get LLM model name."""
        return self.llm_config["model_name"]
    
    @property
    def llm_base_url(self) -> str:
        """Get LLM base URL."""
        return self.llm_config["base_url"]
    
    @property
    def embedder_model_name(self) -> str:
        """Get embedder model name."""
        return self.embedder_config["model_name"]
    
    @property
    def embedder_base_url(self) -> str:
        """Get embedder base URL."""
        return self.embedder_config["base_url"]
    
    def get_llm_config(self) -> Dict[str, str]:
        """
        Get LLM configuration as a dictionary.
        
        Returns:
            Dictionary with 'model_name' and 'base_url'
        """
        return {
            "model_name": self.llm_model_name,
            "base_url": self.llm_base_url
        }
    
    def get_embedder_config(self) -> Dict[str, str]:
        """
        Get embedder configuration as a dictionary.
        
        Returns:
            Dictionary with 'model_name' and 'base_url'
        """
        return {
            "model_name": self.embedder_model_name,
            "base_url": self.embedder_base_url
        }
