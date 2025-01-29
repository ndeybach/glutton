"""Command-line interface for Glutton."""

import glob
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

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


def display_backends() -> None:
    """Display available conversion backends."""
    converter = GluttonConverter()
    table = Table(title="Available Conversion Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Status", style="green")

    for backend in converter.available_backends:
        try:
            # Try to instantiate the backend to check if it's available
            converter._get_backend("test.pdf", backend)
            status = "✓ Available"
        except ImportError:
            status = "✗ Not installed"
        table.add_row(backend, status)

    rprint(Panel.fit(table, title="Conversion Backends"))


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_files: Optional[List[str]] = typer.Argument(
        None,
        help="One or more input files to convert. Glob patterns are supported.",
        show_default=False,
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
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Conversion backend to use (falls back to others if not compatible)",
    ),
    print_output: bool = typer.Option(
        False,
        "--print",
        "-p",
        help="Print output to console",
    ),
    list_backends: bool = typer.Option(
        False,
        "--list-backends",
        "-l",
        help="List available conversion backends and their status",
        rich_help_panel="Backend Management",
    ),
) -> None:
    """Convert files to markdown format using various backends.
    
    If a specific backend is requested but not compatible with the input file,
    Glutton will automatically fall back to other available backends.
    """
    if list_backends:
        display_backends()
        return

    if not input_files:
        ctx.get_help()
        return

    try:
        converter = GluttonConverter(config)
        expanded_files = expand_paths(input_files)

        if len(expanded_files) == 1:
            # Single file conversion
            result = converter.convert(expanded_files[0], output, backend)
            if print_output:
                rprint(result.text_content)
            if result.output_file:
                rprint(f"[green]Converted {result.input_file} to {result.output_file}[/green]")
                if backend:
                    rprint(f"[blue]Using backend: {backend}[/blue]")
        else:
            # Batch conversion
            results = converter.convert_many(expanded_files, output, backend)
            for result in results:
                if print_output:
                    rprint(f"\n[bold]=== {result.input_file} ===[/bold]\n")
                    rprint(result.text_content)
                if result.output_file:
                    rprint(f"[green]Converted {result.input_file} to {result.output_file}[/green]")
                    if backend:
                        rprint(f"[blue]Using backend: {backend}[/blue]")

    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()