# Glutton

A Python utility for converting various file formats to markdown, using multiple conversion backends.

## Features

- Convert various file formats to markdown:
  - Excel files (.xlsx, .xls)
  - PDF documents
  - Word documents (.docx, .doc)
  - PowerPoint presentations
  - HTML files
  - Text files
  - Images
  - Audio (transcription)
  - ZIP archives
- Support for both command-line and programmatic usage
- Flexible output file naming patterns
- Full UTF-8 support
- Batch processing capabilities
- Extensible backend system (currently using markitdown)

## Installation

```bash
# Install the package in development mode
pip install -e .

# For development dependencies (optional)
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bash
# Show help and available options
python -m glutton --help

# Basic conversion with default output naming
python -m glutton input.xlsx

# Specify output file
python -m glutton input.xlsx -o output.md

# Batch processing with pattern
python -m glutton *.xlsx -o {name}_converted.md

# Print output to console
python -m glutton input.xlsx --print

# Use custom configuration
python -m glutton input.xlsx -c glutton.yaml
```

### Programmatic Usage

```python
from glutton import GluttonConverter

# Single file conversion
converter = GluttonConverter()
result = converter.convert("input.xlsx", output="output.md")

# Batch processing
results = converter.convert_many(["file1.xlsx", "file2.xlsx"], 
                               output_pattern="{name}_converted.md")

# Access conversion results
for result in results:
    print(f"Converted {result.input_file} to {result.output_file}")
    print(result.text_content)
```

## Configuration

The converter can be configured through a `glutton.yaml` file in your project directory:

```yaml
output:
  default_pattern: "{name}_md.{ext}"  # Default output file naming pattern
  default_extension: "md"             # Default output file extension

conversion:
  encoding: "utf-8"                   # Input/output file encoding
  table_format: "github"              # Markdown table format style
```

## Output Naming Patterns

The following placeholders are available for output file naming:

- `{name}`: Original filename without extension
- `{ext}`: Output file extension (defaults to 'md')
- `{date}`: Current date (YYYY-MM-DD)
- `{time}`: Current time (HHMMSS)

## Backend System

Glutton currently uses the markitdown library as its primary conversion backend. The architecture is designed to support additional backends in the future, allowing for flexible conversion strategies based on file type and requirements.

## License

MIT License - see LICENSE file for details.