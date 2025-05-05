# AWEsomeScoring

A command line tool for automated writing evaluation (AWE) using different AI models (OpenAI, Claude, Mistral).

## Overview

AWEsomeScoring is a tool that enables benchmarking and evaluation of text using different AI models. It processes text corpora, applies specific prompts to evaluate the texts, and generates standardized outputs for analysis.

## Features

- **Text Corpus Management**: Read, process, and convert text corpora between formats
- **Benchmark Processing**: Evaluate texts using different AI services and models
- **Flexible Configuration**: Configure through command line, config files, or environment variables
- **Multi-model Support**: Leverage OpenAI, Claude, and Mistral models for text evaluation
- **Efficient Processing**: Parallel processing with configurable concurrency
- **Robust Error Handling**: Retry mechanisms and error recovery for API calls

## Installation

### Prerequisites

- Python 3.7 or higher
- API keys for OpenAI, Claude, and/or Mistral (depending on which services you want to use)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AWEsomeScoring.git
cd AWEsomeScoring
```

## Usage

AWEsomeScoring provides a command-line interface with multiple subcommands:

### Configuration

```bash
# Initialize default configuration
awescore config init --output my_config.yaml

# Show current configuration
./AWEsomeScoring config show --config my_config.yaml
```

### Corpus Management

```bash
# Convert corpus between formats
./AWEsomeScoring corpus convert --input texts.txt --output texts.csv --type narrative

# List corpus entries
./AWEsomeScoring corpus list --input texts.csv --limit 10
```

### Running Benchmarks

```bash
# Run benchmark with OpenAI, Claude, and Mistral
./AWEsomeScoring benchmark run \
  --input texts.csv \
  --output results_dir \
  --type narrative \
  --system-prompt system_prompt.txt \
  --user-prompt user_prompt.txt \
  --services openai claude mistral \
  --runs 3 \
  --temperature 0.7
```

## Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_claude_key
export MISTRAL_API_KEY=your_mistral_key
```

## Configuration File

You can create a configuration file (YAML, JSON, or INI format) with the following structure:

```yaml
# API keys (can be overridden by environment variables)
api_openai: ''
api_claude: ''
api_mistral: ''

# Model selection
model_openai: 'gpt-4o-2024-11-20'
model_claude: 'claude-3-7-sonnet-20250219'
model_mistral: 'mistral-large-2411'

# Generation settings
temperature: 0.7

# Performance settings
max_workers: 10
retry_max_attempts: 12
retry_initial_wait: 2
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
