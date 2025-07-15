"""
Command-line argument parser for AWEsomeScoring.

This module contains all argument parser setup functions.
"""

import argparse
from config import CONFIG_VERSION

# Import command implementations
from .config_commands import init_config, show_config, dump_config, validate_config
from .corpus_commands import convert_corpus, list_corpus
from .benchmark_commands import run_benchmark


def add_config_subparsers(subparsers):
    """Add configuration-related subcommands."""
    config_parser = subparsers.add_parser('config', help='Configuration commands')
    config_subparsers = config_parser.add_subparsers(dest="config_cmd", help='Configuration commands')
    
    # Initialize config
    init_parser = config_subparsers.add_parser('init', help='Initialize default configuration file')
    init_parser.add_argument('--output', '-o', default='awesome_config.yaml', 
                            help='Output file path for configuration')
    init_parser.set_defaults(func=init_config)
    
    # Show current config
    show_parser = config_subparsers.add_parser('show', help='Show current configuration')
    show_parser.add_argument('--config', '-c', help='Path to configuration file')
    show_parser.add_argument('--mask-keys', '-m', action='store_true', 
                            help='Mask API keys in the output')
    show_parser.set_defaults(func=show_config)

    # Dump config
    dump_parser = config_subparsers.add_parser('dump', help='Dump the current effective configuration to a file')
    dump_parser.add_argument('--output', '-o', required=True, help='Output file path for configuration dump')
    dump_parser.add_argument('--config', '-c', help='Path to configuration file to use as base')
    dump_parser.add_argument('--format', '-f', choices=['yaml', 'json', 'ini'], default='yaml',
                           help='Output format (yaml, json, or ini)')
    dump_parser.set_defaults(func=dump_config)

    # Validate config
    validate_parser = config_subparsers.add_parser('validate', help='Validate a configuration file')
    validate_parser.add_argument('--config', '-c', required=True, help='Path to configuration file to validate')
    validate_parser.add_argument('--command', help='Validate for a specific command (e.g., benchmark:run)')
    validate_parser.set_defaults(func=validate_config)


def add_corpus_subparsers(subparsers):
    """Add corpus-related subcommands."""
    corpus_parser = subparsers.add_parser('corpus', help='Corpus commands')
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_cmd", help='Corpus commands')
    
    # Convert corpus
    convert_parser = corpus_subparsers.add_parser('convert', help='Convert corpus between formats')
    convert_parser.add_argument('--input', '-i', help='Input corpus file path')
    convert_parser.add_argument('--output', '-o', help='Output file path for converted corpus')
    convert_parser.add_argument('--type', '-t', choices=['basch_narrative', 'basch_instructive'], help='Type of corpus')
    convert_parser.add_argument('--config', '-c', help='Path to configuration file')
    convert_parser.set_defaults(func=convert_corpus)
    
    # List corpus
    list_parser = corpus_subparsers.add_parser('list', help='List all texts in corpus')
    list_parser.add_argument('--input', '-i', help='Input corpus file path')
    list_parser.add_argument('--type', '-t', choices=['basch_narrative', 'basch_instructive'], 
                            help='Type of corpus (not needed for CSV files)')
    list_parser.add_argument('--limit', '-l', type=int, default=5, help='Limit the number of texts to display')
    list_parser.add_argument('--config', '-c', help='Path to configuration file')
    list_parser.set_defaults(func=list_corpus)


def add_benchmark_subparsers(subparsers):
    """Add benchmark-related subcommands."""
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark commands')
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_cmd", help='Benchmark commands')
    
    # Run benchmark
    run_parser = benchmark_subparsers.add_parser('run', help='Run benchmarks on corpus')
    run_parser.add_argument('--input', '-i', help='Input corpus file path')
    run_parser.add_argument('--output', '-o', help='Output directory for results')
    run_parser.add_argument('--type', '-t', choices=['basch_narrative', 'basch_instructive'], help='Type of corpus')
    # Base config
    run_parser.add_argument('--system-prompt', '-s', help='System prompt file path')
    run_parser.add_argument('--user-prompt', '-u', help='User prompt file path')
    run_parser.add_argument('--services', nargs="+", choices=["openai", "claude", "mistral"], default=[], help='AI services to use')
    run_parser.add_argument('--config', '-c', help='Path to configuration file')
    # Config details
    run_parser.add_argument('--runs', '-r', type=int, help='Number of benchmark runs')
    run_parser.add_argument('--limit', '-l', type=int, help='Limit the number of texts to score')
    run_parser.add_argument('--temperature', type=float, help='Temperature for generation')
    run_parser.add_argument('--no-temperature', action='store_true', 
                           help="Don't include temperature in API calls (use model defaults)")
    # Batch processing commands
    run_parser.add_argument('--batch', action='store_true', help='Use batch API mode')
    run_parser.add_argument('--batch-size', type=int, help='Batch size for batch API calls')
    run_parser.add_argument('--batch-poll-interval', type=int, help='Polling interval in seconds for batch API status checks')
    # Other
    run_parser.add_argument('--verbose', '-v', action='count', default=0, 
                           help='Increase verbosity (can be used multiple times)')
    run_parser.set_defaults(func=run_benchmark)


def create_parser():
    """
    Create and return the main argument parser.
    
    Returns:
        argparse.ArgumentParser: The configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=f"AWEsomeScoring (v{CONFIG_VERSION}) - Command line tool for automated writing evaluation (AWE) using different AI models."
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add commands from modules
    add_config_subparsers(subparsers)
    add_corpus_subparsers(subparsers)
    add_benchmark_subparsers(subparsers)

    # Parser common arguments that apply to multiple commands
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--version', '-v', action='store_true', help='Show version information')

    return parser
