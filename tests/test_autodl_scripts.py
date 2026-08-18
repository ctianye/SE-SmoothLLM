import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
AUTODL_DIR = PROJECT_ROOT / "scripts" / "autodl"


def test_vllm_version_is_pinned_once_and_matches_config() -> None:
    config = json.loads(
        (PROJECT_ROOT / "benchmarks" / "configs" / "vicuna_gcg.json").read_text(
            encoding="utf-8"
        )
    )
    requirements = (PROJECT_ROOT / "requirements-autodl.txt").read_text(encoding="utf-8")

    assert f"vllm=={config['serving']['version']}" in requirements


def test_start_script_uses_pinned_model_and_explicit_chat_template() -> None:
    script = (AUTODL_DIR / "start_vllm.sh").read_text(encoding="utf-8")

    assert "model.revision" in script
    assert '--chat-template "${CHAT_TEMPLATE}"' in script
    assert "--generation-config vllm" in script
    assert "--disable-log-requests" not in script
    assert '--host "${HOST}"' in script


def test_vicuna_template_contains_expected_roles_and_generation_marker() -> None:
    template = (AUTODL_DIR / "vicuna_chat_template.jinja").read_text(encoding="utf-8")

    assert "bos_token" in template
    assert "USER: " in template
    assert "ASSISTANT: " in template
    assert "add_generation_prompt" in template


def test_runtime_scripts_fail_on_first_error() -> None:
    for path in AUTODL_DIR.glob("*.sh"):
        content = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in content, path.name
