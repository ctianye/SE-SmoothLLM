# Roadmap

## Package foundation

- Maintain the `src/` package layout and typed extension interfaces.
- Keep CLI, local API, and Python API behavior covered by tests.
- Build both source and wheel distributions in continuous integration.

## Defense implementation

- Add reproducible character-level perturbations used by SmoothLLM baselines.
- Maintain fixed-budget and exact early-stopping paths on one shared executor.
- Preserve fixed-budget verdicts with exhaustive and end-to-end regression tests.
- Extend per-copy traces with exportable run configuration and stopping summaries.

## Backends and deployment

- Validate the OpenAI-compatible backend against a local inference server.
- Add an OpenAI-compatible proxy endpoint for defended chat completions.
- Define timeout, retry, concurrency, and error-reporting behavior.

## Evaluation

- Add reusable JailbreakBench data loading and evaluation commands.
- Report attack success rate, average queries, latency, and early-stop rate.
- Compare undefended, fixed-budget SmoothLLM, and SE-SmoothLLM under identical settings.
- Publish aggregate CSV/JSON results and the commands needed to reproduce them.

## Release

- Add CI for supported Python versions.
- Audit third-party code and dataset licenses in `NOTICE`.
- Publish versioned documentation and a test release to TestPyPI.
