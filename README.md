# AWEsomeScoring

A command line tool for automated writing evaluation (AWE) using different AI models (OpenAI, Claude, Mistral).

## Overview

AWEsomeScoring helps educators and researchers evaluate written texts using AI. It can process entire text corpora, apply customized evaluation prompts, and generate standardized outputs for analysis and comparison across different AI models.

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
- API keys for the services you want to use:
  - [OpenAI API key](https://platform.openai.com/account/api-keys)
  - [Anthropic API key](https://console.anthropic.com/account/keys)
  - [Mistral API key](https://console.mistral.ai/api-keys/)

### Installation (from source)

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

### API Keys

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

## Input Formats

AWEsomeScoring supports two primary corpus formats:

### CSV Format

A simple CSV file with columns:
- `student_id`: Unique identifier for each text
- `text`: The actual text content to evaluate

### Text Formats

Two special formats are supported. They are specified for a research project the author is associated with. 
- `basch_narrative`: Narrative writing samples with numeric student IDs
- `basch_instructive`: Instructive writing samples with text identifiers

## Prompts

You'll need to provide two separate prompt files:

1. **System Prompt**: Instructions for the AI model on how to evaluate texts
2. **User Prompt**: Template for presenting the text to evaluate

The user prompt should include a `{student_text}` placeholder that will be replaced with the actual text to evaluate.

Example system prompt:
```
You are an expert teacher evaluating student writing. Rate each text on a scale of 1-6, where 1 is poor and 6 is excellent. Provide analysis of strengths, weaknesses, and a justification for your score. Format your response as a JSON object with these fields: punktzahl (score), staerken (strengths), schwaechen (weaknesses), and begruendung (justification).
```

Example user prompt:
```
Please evaluate the following student text and provide a score from 1-6, along with an analysis of its strengths and weaknesses:

{student_text}
```

If you use any other curly bracket in your user prompt, you have to use double brackets `{{` and `}}`.

## Results Format

AWEsomeScoring generates a special CSV format (again: for a research project the author is associated with). It alse saves the raw responses from the AI services.

1. **CSV Result Files**: Containing processed results with columns:
   - `student_id`: Text identifier
   - `punktzahl`: Numerical score
   - `staerken`: Strengths analysis
   - `schwaechen`: Weaknesses analysis
   - `begruendung`: Justification for the score

2. **Raw JSON Responses**: Complete API responses from each model

## Advanced Usage

### Controlling Concurrency

```bash
# Run with 5 worker threads instead of the default 10
awescore benchmark run --max-workers 5 [other options]
```

### Multiple Runs for Statistical Analysis

```bash
# Run the benchmark 5 times for statistical significance
awescore benchmark run --runs 5 [other options]
```

### Rate Limiting and Retries

```bash
# Configure retry behavior for API rate limits
awescore benchmark run --retry-max-attempts 15 --retry-initial-wait 3 [other options]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
