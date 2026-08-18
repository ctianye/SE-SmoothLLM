"""在正式实验前对真实 OpenAI 兼容服务执行一次最小请求。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks.jbb_loader import DEFAULT_CONFIG_PATH, load_experiment_config
from benchmarks.run_jbb import (
    _require_int,
    _require_mapping,
    _require_number,
    _require_text,
    probe_model_service,
)
from se_smoothllm.backends.openai_compatible import OpenAICompatibleBackend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="SE_SMOOTHLLM_API_KEY")
    parser.add_argument(
        "--prompt",
        default="Briefly explain why reproducibility matters in experiments.",
    )
    args = parser.parse_args(argv)

    config = load_experiment_config(Path(args.config))
    model = _require_mapping(config, "model")
    generation = _require_mapping(config, "generation")
    serving = _require_mapping(config, "serving")
    base_url = args.base_url or _require_text(serving, "base_url")
    timeout = _require_number(generation, "timeout_seconds")
    models = probe_model_service(base_url, timeout=min(timeout, 10.0))

    backend = OpenAICompatibleBackend(
        base_url=base_url,
        model=_require_text(model, "id"),
        api_key=os.environ.get(args.api_key_env),
        system_prompt=_require_text(model, "system_prompt"),
        temperature=_require_number(generation, "temperature"),
        max_tokens=min(_require_int(generation, "max_tokens"), 32),
        timeout=timeout,
        max_retries=_require_int(generation, "max_retries"),
        retry_backoff=_require_number(generation, "retry_backoff_seconds"),
    )
    result = backend.generate(args.prompt)
    print(
        json.dumps(
            {
                "status": "ok",
                "available_models": models,
                "reported_model": result.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_ms": round(result.latency_ms, 2),
                "response_preview": result.text[:200],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
