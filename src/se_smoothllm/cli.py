"""Command-line interface for SE-SmoothLLM."""

import typer

from se_smoothllm import __version__

app = typer.Typer(help="SE-SmoothLLM command-line tools.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Run SE-SmoothLLM commands."""


@app.command()
def version() -> None:
    """Print the installed package version."""

    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
