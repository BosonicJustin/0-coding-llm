# Coding smoke environment

`coding-smoke` is a six-task, deterministic Prime Verifiers v1 package for checking
that post-training online evaluation and RL plumbing can safely execute and score
Python answers. It is **not** an SFT data source or an SFT requirement: ordinary
offline SFT reads the exported `messages` dataset directly.

The prompts, function names, reference implementations, generators, and oracles in
this package were authored for this repository. No benchmark rows, prompts, tests,
or solutions were imported from MBPP, HumanEval, or another evaluation suite. Keep
this environment out of the frozen final evaluation set; it is an infrastructure
smoke test, not an estimate of general coding quality.

## Safety and leakage properties

- `CodingSmokeTask.NEEDS_CONTAINER = True`; Verifiers refuses its host subprocess
  runtime. Use Docker or Prime Sandbox/VM for every rollout.
- Model-written source reaches only `vf.Runtime.run_uv_script`. The environment host
  never calls `exec`, `eval`, or `subprocess` on it.
- Each task blocks all execution-time network egress and requests bounded CPU,
  memory, disk, and scoring time. The in-runtime verifier adds child CPU, address
  space, output-size, process-count, and wall-time limits.
- Generated cases and expected values are not fields of `CodingSmokeData`, so they
  are not serialized into task payloads or `traces.jsonl`. The verifier remains open
  source; these cases are therefore leakage-resistant plumbing checks, not secrets.
- Reward is binary (`correct`, weight 1.0). Diagnostics report syntax validity,
  entrypoint presence, tests passed/total, pass rate, timeout, runtime error, and
  submitted source size without changing the optimization signal.

The package targets the official Verifiers v1 API inspected at commit
`c51c094a4018471b7fdc873eb5cb55bbd5e956e1`; its in-runtime scoring pattern was
also cross-checked against `PrimeIntellect-ai/prime-envs` commit
`3b2af76ecfd95cf2fde89969d617f025a6d20fa3`.

## Install and validate

From a checkout of Prime Intellect's `verifiers` repository at the commit above:

```bash
uv sync
uv pip install -e /path/to/0-coding-llm/environments/coding_smoke
uv run validate coding-smoke --runtime.type docker
```

For Prime-managed isolation, select the Prime runtime in the eval or training TOML.
Do not override the runtime to `subprocess`; the framework validation and task both
fail closed if that is attempted. Before RL, first run a bounded evaluation and
confirm all six task validations and scorer metrics are healthy.

Use the built-in `null` harness for the smoke evaluation. It gives the model only
the prompt/response interaction, so the verifier prepared in the runtime is not
available through agent shell tools:

```toml
model = "your-served-instruct-checkpoint"
num_tasks = 6
num_rollouts = 1

[env.taskset]
id = "coding-smoke"

[env.agent.harness]
id = "null"

[env.agent.runtime]
type = "docker" # use "prime" on Prime-managed sandboxes
```

Validate it with `uv run eval @ coding-smoke.toml --dry-run`, then run the same
command without `--dry-run`. Do not use the default shell-capable harness for this
taskset: doing so makes an open-source verifier unnecessarily inspectable during
the rollout.
