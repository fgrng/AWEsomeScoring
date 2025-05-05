# Define expected CSV headers for output
CSV_HEADERS = ["student_id", "punktzahl", "staerken", "schwaechen", "begruendung"]

class ConfigManager:
    """
    Manages configuration settings for the benchmark tool.
    
    This class handles loading settings from environment variables,
    config files, and command line arguments, with appropriate precedence.
    """
    
    # Default config file locations to check
    CONFIG_FILES = [
        './benchmark_config.yaml',
        './benchmark_config.yml',
        './benchmark_config.json',
        './benchmark_config.ini',
        '~/.config/benchmark_tool/config.yaml',
        '~/.config/benchmark_tool/config.yml',
        '~/.config/benchmark_tool/config.json',
        '~/.config/benchmark_tool/config.ini'
    ]
    
    # Environment variable names for API keys
    ENV_VARS = {
        'api_openai': 'OPENAI_API_KEY',
        'api_claude': 'ANTHROPIC_API_KEY',
        'api_mistral': 'MISTRAL_API_KEY',
        'org_openai': 'OPENAI_ORG_ID'
    }
    
    @classmethod
    def load_config(cls, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Load configuration from files, environment variables, and command line arguments.
        
        Priority order (highest to lowest):
        1. Command line arguments
        2. Environment variables
        3. Config file specified by --config
        4. Default config files in CONFIG_FILES
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Dictionary with consolidated configuration settings
        """
        config = {}
        
        # Start with default values
        config.update({
            'api_openai': '',
            'api_claude': '',
            'api_mistral': '',
            'org_openai': '',
            'url_openai': 'https://api.openai.com/v1/chat/completions',
            'url_claude': 'https://api.anthropic.com/v1/messages',
            'temperature': 0.7,
            'model_openai': 'gpt-4o-2024-11-20',
            'model_claude': 'claude-3-7-sonnet-20250219',
            'model_mistral': 'mistral-large-2411',
            'comment_openai': 'Run',
            'comment_claude': 'Run',
            'comment_mistral': 'Run',
            'num_runs': 1,
            'max_workers': 10,
            'retry_max_attempts': 12,
            'retry_initial_wait': 2
        })
        
        # Check for default config files
        for config_file in cls.CONFIG_FILES:
            expanded_path = os.path.expanduser(config_file)
            if os.path.exists(expanded_path):
                logger.info(f"Loading configuration from {expanded_path}")
                config.update(cls._load_from_file(expanded_path))
                break
        
        # Check for specified config file
        if hasattr(args, 'config') and args.config:
            if os.path.exists(args.config):
                logger.info(f"Loading configuration from {args.config}")
                config.update(cls._load_from_file(args.config))
            else:
                logger.warning(f"Specified config file not found: {args.config}")
        
        # Apply environment variables
        for key, env_var in cls.ENV_VARS.items():
            if env_var in os.environ:
                config[key] = os.environ[env_var]
        
        # Apply command line arguments (overrides previous settings)
        for key, value in vars(args).items():
            if value is not None:  # Only override if the argument was provided
                # Convert CLI argument name format to config key format
                config_key = key.replace('-', '_')
                
                # Special handling for certain keys
                if key == 'openai_key':
                    config['api_openai'] = value
                elif key == 'claude_key':
                    config['api_claude'] = value
                elif key == 'mistral_key':
                    config['api_mistral'] = value
                elif key == 'openai_org':
                    config['org_openai'] = value
                elif key == 'openai_model':
                    config['model_openai'] = value
                elif key == 'claude_model':
                    config['model_claude'] = value
                elif key == 'mistral_model':
                    config['model_mistral'] = value
                elif key == 'openai_comment':
                    config['comment_openai'] = value
                elif key == 'claude_comment':
                    config['comment_claude'] = value
                elif key == 'mistral_comment':
                    config['comment_mistral'] = value
                elif key == 'runs':
                    config['num_runs'] = value
                else:
                    config[config_key] = value
        
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
                    config = yaml.safe_load(f)
            
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
        
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {str(e)}")
        
        return config


class Config:
    """
    Configuration class to hold and validate settings.
    
    This class stores all configuration settings and validates them
    to ensure they meet requirements before running benchmarks.
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Load configuration from various sources
        config = ConfigManager.load_config(args)
        
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
        
        # Input/output settings
        self.input_corpus_path = config.get('input', '')
        self.output_dir = config.get('output', f"{datetime.now().strftime('%Y%m%d')}_Benchmark_Output")
        
        # System and user prompts
        self.system_prompt_path = config.get('system_prompt', '')
        self.user_prompt_path = config.get('user_prompt', '')
        
        # Corpus type (narrative or instructive)
        self.corpus_type = config.get('corpus_type', '')
        
        # Services to use
        services = config.get('services', ['openai', 'claude', 'mistral'])
        self.use_openai = 'openai' in services
        self.use_claude = 'claude' in services
        self.use_mistral = 'mistral' in services
        
        # Number of runs
        self.num_runs = config.get('num_runs', 1)
        
        # Concurrency settings
        self.max_workers = config.get('max_workers', 10)
        
        # Retry settings
        self.retry_max_attempts = config.get('retry_max_attempts', 12)
        self.retry_initial_wait = config.get('retry_initial_wait', 2)
        
        # Validate the configuration
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the configuration settings.
        
        Raises:
            ValueError: If any required configuration is missing or invalid
        """
        # Required file paths
        if not self.input_corpus_path:
            raise ValueError("Input corpus file path is required")
            
        if not os.path.exists(self.input_corpus_path):
            raise ValueError(f"Input corpus file does not exist: {self.input_corpus_path}")
        
        if not self.system_prompt_path:
            raise ValueError("System prompt file path is required")
            
        if not os.path.exists(self.system_prompt_path):
            raise ValueError(f"System prompt file does not exist: {self.system_prompt_path}")
        
        if not self.user_prompt_path:
            raise ValueError("User prompt file path is required")
            
        if not os.path.exists(self.user_prompt_path):
            raise ValueError(f"User prompt file does not exist: {self.user_prompt_path}")
        
        # Required corpus type
        if not self.corpus_type:
            raise ValueError("Corpus type is required (narrative or instructive)")
            
        if self.corpus_type not in ['narrative', 'instructive']:
            raise ValueError(f"Invalid corpus type: {self.corpus_type} (must be 'narrative' or 'instructive')")
        
        # Check if any service is enabled
        if not any([self.use_openai, self.use_claude, self.use_mistral]):
            raise ValueError("At least one AI service must be enabled")
        
        # Check if the necessary API keys are available for the requested services
        if self.use_openai and not self.api_openai:
            raise ValueError("OpenAI API key is required for OpenAI service")
        
        if self.use_claude and not self.api_claude:
            raise ValueError("Anthropic API key is required for Claude service")
        
        if self.use_mistral and not self.api_mistral:
            raise ValueError("Mistral API key is required for Mistral service")
        
        # Validate numeric settings
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temperature}")
            
        if self.num_runs < 1:
            raise ValueError(f"Number of runs must be at least 1, got {self.num_runs}")
            
        if self.max_workers < 1:
            raise ValueError(f"Maximum workers must be at least 1, got {self.max_workers}")
            
        if self.retry_max_attempts < 1:
            raise ValueError(f"Maximum retry attempts must be at least 1, got {self.retry_max_attempts}")
            
        if self.retry_initial_wait < 0.1:
            raise ValueError(f"Initial retry wait time must be at least 0.1 seconds, got {self.retry_initial_wait}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)