"""Command-line interface for Glutton."""

import glob
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint

from glutton.converter import GluttonConverter

app = typer.Typer(
    help="Convert various file formats to markdown using multiple backends.",
    add_completion=False,
)


def expand_paths(paths: List[str]) -> List[str]:
    """Expand glob patterns in paths."""
    expanded = []
    for path in paths:
        if "*" in path:
            expanded.extend(glob.glob(path))
        else:
            expanded.append(path)
    return expanded


@app.command()
def convert(
    input_files: List[str] = typer.Argument(
        ...,
        help="One or more input files to convert. Glob patterns are supported.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file or pattern. For multiple files, use patterns like {name}_converted.md",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    print_output: bool = typer.Option(
        False,
        "--print",
        "-p",
        help="Print output to console",
    ),
) -> None:
    """Convert files to markdown format."""
    try:
        converter = GluttonConverter(config)
        expanded_files = expand_paths(input_files)

        if len(expanded_files) == 1:
            # Single file conversion
            result = converter.convert(expanded_files[0], output)
            if print_output:
                rprint(result.text_content)
            if result.output_file:
                rprint(f"[green]Converted {result.input_file} to {result.output_file}[/green]")
        else:
            # Batch conversion
            results = converter.convert_many(expanded_files, output)
            for result in results:
                if print_output:
                    rprint(f"\n[bold]=== {result.input_file} ===[/bold]\n")
                    rprint(result.text_content)
                if result.output_file:
                    rprint(f"[green]Converted {result.input_file} to {result.output_file}[/green]")

    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()