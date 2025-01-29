"""Example usage of Glutton converter."""

import sys
from pathlib import Path
from typing import List, Optional

from glutton import GluttonConverter


def convert_files(
    files: List[str], output_pattern: Optional[str] = None, config_file: Optional[str] = None
) -> None:
    """Convert files using programmatic API."""
    converter = GluttonConverter(config_file)
    
    if len(files) == 1:
        # Single file conversion
        result = converter.convert(files[0], output_pattern)
        print(f"Converted {result.input_file}")
        print("\nOutput content:")
        print(result.text_content)
    else:
        # Batch conversion
        results = converter.convert_many(files, output_pattern)
        for result in results:
            print(f"Converted {result.input_file}")
            if result.output_file:
                print(f"Output saved to: {result.output_file}")


if __name__ == "__main__":
    # Example of programmatic usage
    if len(sys.argv) > 1:
        # If arguments provided, use them
        files = sys.argv[1:]
        convert_files(files, "{name}_converted.md")
    else:
        # Demo usage with example file
        example_file = Path("test.xlsx")
        if example_file.exists():
            convert_files([str(example_file)])
        else:
            print(f"Please create {example_file} or provide input files as arguments")
            print("\nUsage examples:")
            print("  python main.py input.xlsx")
            print("  python main.py file1.xlsx file2.xlsx")
            print("\nFor more options, use the CLI:")
            print("  python -m glutton --help")