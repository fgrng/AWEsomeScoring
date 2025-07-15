"""
Configuration management for AWEsomeScoring.

This module handles loading, validating, and accessing configuration settings
from various sources (environment variables, config files, command line arguments).
"""

import argparse
import json
import logging
import os
import sys
from configparser import ConfigParser
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Type, Tuple
from enum import Enum

logger = logging.getLogger("AWEsomeScoring")

## Configuration schema version
CONFIG_VERSION = "1.0.0"

## Define expected CSV headers for output
CSV_HEADERS = ["student_id", "punktzahl", "staerken", "schwaechen", "begruendung"]


class ConfigSource(Enum):
    """Enum representing the source of a configuration value."""
    DEFAULT = "default"
    CONFIG_FILE = "config_file"
    ENV_VAR = "environment_variable"
    CLI = "command_line"


class ConfigSchema:
    """
    Defines the schema for configuration validation.
    
    This class specifies the expected types, allowed values, and requirements
    for each configuration option.
    """
    
    # Schema definition
    SCHEMA = {
        # API Keys and URLs
        "api_openai": {"type": str, "required": False, "default": ""},
        "api_claude": {"type": str, "required": False, "default": ""},
        "api_mistral": {"type": str, "required": False, "default": ""},
        "org_openai": {"type": str, "required": False, "default": ""},
        "url_openai": {"type": str, "required": False, "default": "https://api.openai.com/v1/chat/completions"},
        "url_claude": {"type": str, "required": False, "default": "https://api.anthropic.com/v1/messages"},
        
        # Generation settings
        "temperature": {"type": float, "required": False, "default": 0.7, "min": 0.0, "max": 2.0},
        "use_temperature": {"type": bool, "required": False, "default": True},
        
        # Model selections
        "model_openai": {"type": str, "required": False, "default": "gpt-4o-2024-11-20"},
        "model_claude": {"type": str, "required": False, "default": "claude-3-7-sonnet-20250219"},
        "model_mistral": {"type": str, "required": False, "default": "mistral-large-2411"},
        
        # Run comments (used in filenames)
        "comment_openai": {"type": str, "required": False, "default": "Run"},
        "comment_claude": {"type": str, "required": False, "default": "Run"},
        "comment_mistral": {"type": str, "required": False, "default": "Run"},
        
        # Service toggles
        "use_openai": {"type": bool, "required": False, "default": False},
        "use_claude": {"type": bool, "required": False, "default": False},
        "use_mistral": {"type": bool, "required": False, "default": False},
        
        # File paths
        "input_corpus_path": {"type": str, "required": True, "default": ""},
        "output_dir": {"type": str, "required": False, "default": ""},
        "system_prompt_path": {"type": str, "required": True, "default": ""},
        "user_prompt_path": {"type": str, "required": True, "default": ""},
        
        # Corpus type
        "corpus_type": {"type": str, "required": True, "default": "", 
                       "allowed": ["basch_narrative", "basch_instructive"]},
        
        # Performance settings
        "num_runs": {"type": int, "required": False, "default": 1, "min": 1},
        "limit": {"type": int, "required": False, "default": 0, "min": 0},
        "max_workers": {"type": int, "required": False, "default": 10, "min": 1},
        "retry_max_attempts": {"type": int, "required": False, "default": 12, "min": 1},
        "retry_initial_wait": {"type": float, "required": False, "default": 2.0, "min": 0.1},

        # AI Batch Settings
        "use_batch_mode": {"type": bool, "required": False, "default": True},
        "batch_size": {"type": int, "required": False, "default": 0, "min": 0},
        "batch_poll_interval": {"type": float, "required": False, "default": 60.0, "min": 0.1},        
        
        # Version information
        "config_version": {"type": str, "required": False, "default": CONFIG_VERSION},
    }
    
    @classmethod
    def validate(cls, config: Dict[str, Any], for_command: str = None) -> Tuple[bool, List[str]]:
        """
        Validate a configuration dictionary against the schema.
        
        Args:
            config: Configuration dictionary to validate
            for_command: Command for which to validate (some commands have different requirements)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Skip extensive validation for 'config' commands
        if for_command and for_command.startswith('config:'):
            # Only do minimal validation for config commands
            return True, []
        
        # Check for required fields based on command
        required_fields = []
        if for_command == "benchmark:run":
            # For benchmark run, all these fields are required
            required_fields = ["input_corpus_path", "system_prompt_path", "user_prompt_path", "corpus_type"]
            
            # Also check if at least one service is enabled
            if not any([config.get("use_openai"), config.get("use_claude"), config.get("use_mistral")]):
                errors.append("At least one AI service must be enabled (use_openai, use_claude, or use_mistral)")
                
            # Check for API keys for enabled services
            if config.get("use_openai") and not config.get("api_openai"):
                errors.append("OpenAI API key is required when OpenAI service is enabled")
            if config.get("use_claude") and not config.get("api_claude"):
                errors.append("Claude API key is required when Claude service is enabled")
            if config.get("use_mistral") and not config.get("api_mistral"):
                errors.append("Mistral API key is required when Mistral service is enabled")
                
        # Validate each field against schema
        for key, schema in cls.SCHEMA.items():
            # Skip if field is not present and not required
            if key not in config:
                if key in required_fields or schema.get("required"):
                    errors.append(f"Required field '{key}' is missing")
                continue
                
            value = config[key]
            
            # Check type
            expected_type = schema.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"Field '{key}' has wrong type: expected {expected_type.__name__}, got {type(value).__name__}")
                continue
                
            # Check allowed values
            allowed_values = schema.get("allowed")
            if allowed_values and value not in allowed_values:
                errors.append(f"Field '{key}' has invalid value: {value}. Allowed values: {allowed_values}")
                
            # Check min/max values for numeric types
            if isinstance(value, (int, float)):
                min_val = schema.get("min")
                if min_val is not None and value < min_val:
                    errors.append(f"Field '{key}' is too small: {value}. Minimum: {min_val}")
                    
                max_val = schema.get("max")
                if max_val is not None and value > max_val:
                    errors.append(f"Field '{key}' is too large: {value}. Maximum: {max_val}")
                    
        # File existence checks for path fields
        path_fields = ["input_corpus_path", "system_prompt_path", "user_prompt_path"]
        for field in path_fields:
            if field in config and config[field] and for_command != "config:validate":
                if not os.path.exists(config[field]):
                    errors.append(f"File specified in '{field}' does not exist: {config[field]}")
                    
        return len(errors) == 0, errors
    

class ConfigManager:
    """
    Manages configuration settings for the benchmark tool.
    
    This class handles loading settings from environment variables,
    config files, and command line arguments, with appropriate precedence.
    """
    
    # Default config file locations to check
    CONFIG_FILES = [
        './awesome_config.yaml',
        './awesome_config.yml',
        './awesome_config.json',
        './awesome_config.ini',
        '~/.config/awesome_scoring/config.yaml',
        '~/.config/awesome_scoring/config.yml',
        '~/.config/awesome_scoring/config.json',
        '~/.config/awesome_scoring/config.ini'
    ]
    
    # Environment variable names for API keys
    ENV_VARS = {
        'api_openai': 'OPENAI_API_KEY',
        'api_claude': 'ANTHROPIC_API_KEY',
        'api_mistral': 'MISTRAL_API_KEY',
        'org_openai': 'OPENAI_ORG_ID'
    }
    
    # Mapping between CLI argument names and internal config names
    KEY_MAPPINGS = {
        'input': 'input_corpus_path',
        'output': 'output_dir',
        'system-prompt': 'system_prompt_path',
        'user-prompt': 'user_prompt_path',
        'type': 'corpus_type',
        'runs': 'num_runs',
        'services': 'services',
        'openai_key': 'api_openai',
        'claude_key': 'api_claude',
        'mistral_key': 'api_mistral',
        'openai_org': 'org_openai',
        'openai_model': 'model_openai',
        'claude_model': 'model_claude',
        'mistral_model': 'model_mistral',
        'openai_comment': 'comment_openai',
        'claude_comment': 'comment_claude',
        'mistral_comment': 'comment_mistral',
    }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            Dictionary with default configuration settings
        """
        return {field: schema['default'] for field, schema in ConfigSchema.SCHEMA.items()}
    
    @classmethod
    def load_config(cls, args: argparse.Namespace, command: str = None) -> Dict[str, Any]:
        """
        Load configuration from files, environment variables, and command line arguments.
        
        Priority order (highest to lowest):
        1. Command line arguments
        2. Environment variables
        3. Config file specified by --config
        4. Default config files in CONFIG_FILES
        
        Args:
            args: Parsed command line arguments
            command: Current command being executed (for validation)
            
        Returns:
            Dictionary with consolidated configuration settings
        """
        # Start with default values
        config = cls.get_default_config()
        
        # Track the source of each configuration value for diagnostics
        config_sources = {k: ConfigSource.DEFAULT for k in config}
        
        # Load from default config files
        loaded_from_default = False
        for config_file in cls.CONFIG_FILES:
            expanded_path = os.path.expanduser(config_file)
            if os.path.exists(expanded_path):
                logger.info(f"Loading configuration from {expanded_path}")
                file_config = cls._load_from_file(expanded_path)
                if file_config:
                    for k, v in file_config.items():
                        config[k] = v
                        config_sources[k] = ConfigSource.CONFIG_FILE
                    config['_config_file_path'] = expanded_path
                    loaded_from_default = True
                    break
        
        # Load from specified config file (higher precedence)
        if hasattr(args, 'config') and args.config:
            if os.path.exists(args.config):
                logger.info(f"Loading configuration from {args.config}")
                file_config = cls._load_from_file(args.config)
                if file_config:
                    for k, v in file_config.items():
                        config[k] = v
                        config_sources[k] = ConfigSource.CONFIG_FILE
                    config['_config_file_path'] = args.config
            else:
                logger.warning(f"Specified config file not found: {args.config}")
        
        # Apply environment variables (higher precedence)
        for key, env_var in cls.ENV_VARS.items():
            if env_var in os.environ:
                config[key] = os.environ[env_var]
                config_sources[key] = ConfigSource.ENV_VAR
        
        # Apply command line arguments (highest precedence)
        for key, value in vars(args).items():
            if value is not None:  # Only override if the argument was provided
                # Use the key mapping if available
                config_key = cls.KEY_MAPPINGS.get(key, key.replace('-', '_'))
                
                # Special handling for services list
                if key == 'services' and value:
                    # Enable services based on the list
                    for service in value:
                        service_key = f"use_{service}"
                        config[service_key] = True
                        config_sources[service_key] = ConfigSource.CLI
                else:
                    # Set the value in the config
                    config[config_key] = value
                    config_sources[config_key] = ConfigSource.CLI
        
        # Store the sources for diagnostics
        config['_sources'] = {k: v.value for k, v in config_sources.items()}
        
        # Run schema validation
        is_valid, errors = ConfigSchema.validate(config, command)
        if not is_valid and command and not command.startswith('config:'):
            error_msg = "Configuration validation failed:\n" + "\n".join([f"- {err}" for err in errors])
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return config
    
    @staticmethod
    def _load_from_file(file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file based on its extension.
        
        Supports YAML, JSON, and INI formats.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Dictionary with configuration settings
        """
        config = {}
        
        try:
            if file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
            
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    config = json.load(f)
            
            elif file_path.endswith('.ini'):
                parser = ConfigParser()
                parser.read(file_path)
                
                # Convert ConfigParser object to dictionary
                for section in parser.sections():
                    for key, value in parser.items(section):
                        # Try to convert string values to appropriate types
                        if value.isdigit():
                            value = int(value)
                        elif value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        elif value.replace('.', '', 1).isdigit():
                            value = float(value)
                            
                        config[key] = value
                        
            # Ensure the config contains version information
            if config and 'config_version' not in config:
                config['config_version'] = CONFIG_VERSION
                
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {str(e)}")
            return {}
        
        return config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], file_path: str) -> bool:
        """
        Save configuration to a file.
        
        Supported formats: YAML, JSON, and INI.
        
        Args:
            config: Configuration dictionary to save
            file_path: Path to save the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Filter out internal keys
            filtered_config = {k: v for k, v in config.items() if not k.startswith('_')}
            
            # Ensure version information is included
            if 'config_version' not in filtered_config:
                filtered_config['config_version'] = CONFIG_VERSION
            
            # Save based on file extension
            if file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'w') as f:
                    yaml.dump(filtered_config, f, default_flow_style=False)
            
            elif file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(filtered_config, f, indent=2)
            
            elif file_path.endswith('.ini'):
                parser = ConfigParser()
                parser.add_section('DEFAULT')
                
                for key, value in filtered_config.items():
                    parser.set('DEFAULT', key, str(value))
                
                with open(file_path, 'w') as f:
                    parser.write(f)
            
            else:
                # Default to YAML if extension is not recognized
                with open(file_path, 'w') as f:
                    yaml.dump(filtered_config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {str(e)}")
            return False
    
    @classmethod
    def get_config_source_description(cls, config: Dict[str, Any]) -> str:
        """
        Get a human-readable description of configuration sources.
        
        Args:
            config: Configuration dictionary with source information
            
        Returns:
            Formatted string describing configuration sources
        """
        if '_sources' not in config:
            return "Configuration source information not available"
        
        sources = config['_sources']
        
        # Count occurrences of each source
        source_counts = {}
        for source in sources.values():
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Build description
        lines = ["Configuration values came from:"]
        
        if source_counts.get(ConfigSource.CLI.value, 0) > 0:
            lines.append(f"- Command line: {source_counts.get(ConfigSource.CLI.value, 0)} values")
            
        if source_counts.get(ConfigSource.ENV_VAR.value, 0) > 0:
            lines.append(f"- Environment variables: {source_counts.get(ConfigSource.ENV_VAR.value, 0)} values")
            
        if source_counts.get(ConfigSource.CONFIG_FILE.value, 0) > 0:
            config_file = config.get('_config_file_path', 'unknown')
            lines.append(f"- Config file ({config_file}): {source_counts.get(ConfigSource.CONFIG_FILE.value, 0)} values")
            
        if source_counts.get(ConfigSource.DEFAULT.value, 0) > 0:
            lines.append(f"- Default values: {source_counts.get(ConfigSource.DEFAULT.value, 0)} values")
            
        return "\n".join(lines)


class Config:
    """
    Configuration class to hold and validate settings.
    
    This class stores all configuration settings and validates them
    to ensure they meet requirements before running benchmarks.
    """
    
    def __init__(self, args: argparse.Namespace, command: str = None):
        """
        Initialize configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
            command: Current command being executed (for validation)
        """
        # Load configuration from various sources
        config = ConfigManager.load_config(args, command)
        
        # Store the original config for reference
        self._original_config = config.copy()
        
        # Store command for which this config was created
        self._command = command
        
        # API Keys
        self.api_openai = config.get('api_openai', '')
        self.api_claude = config.get('api_claude', '')
        self.api_mistral = config.get('api_mistral', '')
        
        # API URLs
        self.url_openai = config.get('url_openai', 'https://api.openai.com/v1/chat/completions')
        self.url_claude = config.get('url_claude', 'https://api.anthropic.com/v1/messages')
        
        # Organization IDs
        self.org_openai = config.get('org_openai', '')
        
        # Temperature setting
        self.temperature = config.get('temperature', 0.7)
        self.use_temperature = config.get('use_temperature', True)
        
        # Models to use
        self.model_openai = config.get('model_openai', 'gpt-4o-2024-11-20')
        self.model_claude = config.get('model_claude', 'claude-3-7-sonnet-20250219')
        self.model_mistral = config.get('model_mistral', 'mistral-large-2411')
        
        # Comments for file naming
        self.comment_openai = config.get('comment_openai', 'Run')
        self.comment_claude = config.get('comment_claude', 'Run')
        self.comment_mistral = config.get('comment_mistral', 'Run')
        
        # Input/output settings - use standardized names
        self.input_corpus_path = config.get('input_corpus_path', '')
        self.output_dir = config.get('output_dir', f"{datetime.now().strftime('%Y%m%d')}_Benchmark_Output")
        
        # System and user prompts - use standardized names
        self.system_prompt_path = config.get('system_prompt_path', '')
        self.user_prompt_path = config.get('user_prompt_path', '')
        
        # Corpus type
        self.corpus_type = config.get('corpus_type', '')
        
        # Services to use
        self.use_openai = config.get('use_openai', False)
        self.use_claude = config.get('use_claude', False)
        self.use_mistral = config.get('use_mistral', False)
        
        # Number of runs
        self.num_runs = config.get('num_runs', 1)

        # Limit of student texts
        self.limit = config.get('limit', 0)
        
        # Concurrency settings
        self.max_workers = config.get('max_workers', 10)
        
        # Retry settings
        self.retry_max_attempts = config.get('retry_max_attempts', 12)
        self.retry_initial_wait = config.get('retry_initial_wait', 2)

        # Batch processing settings
        self.use_batch_mode = config.get('use_batch_mode', False)
        self.batch_size = config.get('batch_size', 50)
        self.batch_poll_interval = config.get('batch_poll_interval', 60)

        if self.limit < 0:
            raise ValueError(
                f"Limit of students text must be at least 1 or 0 for no limit, got {self.limit}"
            )
            
        if self.max_workers < 1:
            raise ValueError(f"Maximum workers must be at least 1, got {self.max_workers}")
            
        if self.retry_max_attempts < 1:
            raise ValueError(f"Maximum retry attempts must be at least 1, got {self.retry_max_attempts}")
            
        if self.retry_initial_wait < 0.1:
            raise ValueError(f"Initial retry wait time must be at least 0.1 seconds, got {self.retry_initial_wait}")
        
        # Version information
        self.config_version = config.get('config_version', CONFIG_VERSION)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {key: value for key, value in vars(self).items() 
                if not key.startswith('_') and not callable(value)}
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        return ConfigSchema.validate(self.to_dict(), self._command)
    
    def get_source_description(self) -> str:
        """
        Get a human-readable description of configuration sources.
        
        Returns:
            Formatted string describing configuration sources
        """
        return ConfigManager.get_config_source_description(self._original_config)
