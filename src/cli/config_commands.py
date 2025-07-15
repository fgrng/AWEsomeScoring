"""
Configuration command implementations for AWEsomeScoring CLI.

This module contains all config-related command implementations.
"""

import logging
import yaml
from config import ConfigManager, Config, ConfigSchema

logger = logging.getLogger("awesome_scoring")


def init_config(args):
    """Initialize a default configuration file."""
    try:
        # Get default configuration
        default_config = ConfigManager.get_default_config()
        
        # Save the configuration to the specified file
        success = ConfigManager.save_config(default_config, args.output)
        
        if success:
            logger.info(f"Configuration initialized at {args.output}")
            
            # Print environment variable settings advice
            print("\nTo set API keys as environment variables:")
            print("export OPENAI_API_KEY=your_key_here")
            print("export ANTHROPIC_API_KEY=your_key_here")
            print("export MISTRAL_API_KEY=your_key_here")
            
            # Print next steps
            print("\nNext steps:")
            print(f"1. Edit {args.output} to configure your settings")
            print("2. Run 'awescore config validate --config your_config.yaml' to validate your configuration")
            print("3. Use your configuration with other commands via --config your_config.yaml")
        else:
            logger.error(f"Failed to initialize configuration at {args.output}")
        
    except Exception as e:
        logger.error(f"Error creating config file: {str(e)}")
        print(f"\nError: {str(e)}")


def show_config(args):
    """Show current configuration."""
    try:
        # Create Config object that combines all configuration sources
        config_obj = Config(args, command="config:show")
        
        # Get dictionary representation of the configuration
        config = config_obj.to_dict()
        
        # Mask API keys if requested
        if getattr(args, 'mask_keys', False):
            for key in config:
                if 'api_' in key and config[key]:
                    config[key] = config[key][:4] + '...' + config[key][-4:] if len(config[key]) > 8 else "****"
        
        # Get the configuration source description
        source_description = config_obj.get_source_description()
        
        # Display configuration information
        print("\nCurrent effective configuration:")
        print(yaml.dump(config, default_flow_style=False))
        
        # Show configuration sources
        print("\n" + source_description)
        
        # Show information about config file if available
        if hasattr(config_obj, '_original_config') and '_config_file_path' in config_obj._original_config:
            print(f"\nConfig file used: {config_obj._original_config['_config_file_path']}")
        
        # Show configuration version
        print(f"\nConfiguration schema version: {config_obj.config_version}")
    
    except Exception as e:
        logger.error(f"Error showing configuration: {str(e)}")
        print(f"\nError: {str(e)}")


def dump_config(args):
    """Dump the current effective configuration to a file."""
    try:
        # Create Config object that combines all configuration sources
        config_obj = Config(args, command="config:dump")
        
        # Get dictionary representation of the configuration
        config = config_obj.to_dict()
        
        # Adjust output file name based on specified format
        output_path = args.output
        if args.format == 'yaml' and not output_path.endswith(('.yaml', '.yml')):
            output_path += '.yaml'
        elif args.format == 'json' and not output_path.endswith('.json'):
            output_path += '.json'
        elif args.format == 'ini' and not output_path.endswith('.ini'):
            output_path += '.ini'
        
        # Save configuration to file
        success = ConfigManager.save_config(config, output_path)
        
        if success:
            logger.info(f"Configuration dumped to {output_path}")
            print(f"\nConfiguration successfully dumped to {output_path}")
            
            # Show the configuration source description
            print("\n" + config_obj.get_source_description())
        else:
            logger.error(f"Failed to dump configuration to {output_path}")
            print(f"\nError: Failed to dump configuration to {output_path}")
    
    except Exception as e:
        logger.error(f"Error dumping configuration: {str(e)}")
        print(f"\nError: {str(e)}")


def validate_config(args):
    """Validate a configuration file."""
    try:
        # Load the configuration from the specified file
        file_config = ConfigManager._load_from_file(args.config)
        
        if not file_config:
            print(f"\nError: Could not load configuration from {args.config}")
            return
        
        # Validate the configuration
        command = args.command if args.command else None
        is_valid, errors = ConfigSchema.validate(file_config, command)
        
        # Display validation results
        if is_valid:
            print(f"\nConfiguration file {args.config} is valid")
            if command:
                print(f"Configuration is valid for command: {command}")
            print(f"Configuration schema version: {file_config.get('config_version', 'Not specified')}")
        else:
            print(f"\nConfiguration file {args.config} has validation errors:")
            for i, error in enumerate(errors):
                print(f"{i+1}. {error}")
            
            print("\nPlease fix these errors and try again.")
    
    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        print(f"\nError: {str(e)}")
