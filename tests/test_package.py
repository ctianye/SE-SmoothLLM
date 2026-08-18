from typer.testing import CliRunner

import se_smoothllm
from se_smoothllm.cli import app


def test_package_import_exposes_version() -> None:
    assert se_smoothllm.__version__ == "0.1.0"


def test_cli_prints_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == se_smoothllm.__version__
