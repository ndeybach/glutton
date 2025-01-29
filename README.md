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
- Multiple conversion backends:
  - markitdown (default)
  - docling (optional)
- Automatic backend fallback
- Support for both command-line and programmatic usage
- Flexible output file naming patterns
- Full UTF-8 support
- Batch processing capabilities

## Installation

```bash
# Basic installation with markitdown backend
pip install -e .

# Install with docling backend support
pip install -e ".[docling]"

# Install all optional dependencies (including development tools)
pip install -e ".[all]"
```

## Usage

### Command Line Interface

```bash
# Show help and available options
python -m glutton --help

# List available backends and their status
python -m glutton --list-backends

# Basic conversion with default backend
python -m glutton input.xlsx

# Use specific backend (falls back to others if not compatible)
python -m glutton input.pdf --backend docling

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

# Single file conversion with default backend
converter = GluttonConverter()
result = converter.convert("input.xlsx", output="output.md")

# Use specific backend (falls back to others if not compatible)
result = converter.convert("input.pdf", output="output.md", backend="docling")

# Batch processing
results = converter.convert_many(
    ["file1.xlsx", "file2.pdf"], 
    output_pattern="{name}_converted.md",
    backend="docling"  # Optional backend selection
)

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
  default_backend: "markitdown"       # Default conversion backend

  # Backend-specific settings
  backends:
    markitdown:
      supported_formats: [".xlsx", ".xls", ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".html", ".txt", ".zip"]
    
    docling:
      supported_formats: [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".html"]
```

## Output Naming Patterns

The following placeholders are available for output file naming:

- `{name}`: Original filename without extension
- `{ext}`: Output file extension (defaults to 'md')
- `{date}`: Current date (YYYY-MM-DD)
- `{time}`: Current time (HHMMSS)

## Backend System

Glutton supports multiple conversion backends:

1. markitdown (default):
   - Broad format support
   - Installed by default
   - Handles Excel, PDF, Word, PowerPoint, HTML, text files, and more

2. docling (optional):
   - Specialized in document processing
   - Install with `pip install -e ".[docling]"`
   - Best for PDF, Word, and PowerPoint files

The backend system features automatic fallback: if a requested backend doesn't support the input file type or isn't installed, Glutton will automatically try other available backends.

## License

MIT License - see LICENSE file for details.