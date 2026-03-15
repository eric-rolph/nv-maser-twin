# Contributing to NV Maser Digital Twin

Thank you for contributing! This document explains how to set up your environment, follow project conventions, and get changes merged.

---

## Getting Started

```bash
git clone https://github.com/<your-fork>/nv-maser-twin.git
cd nv-maser-twin

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev,api]"
```

This installs the package in editable mode together with all development dependencies (`pytest`, `pytest-cov`, `httpx`, `jupyter`) and the API extras (`fastapi`, `uvicorn`).

---

## Running Tests

```bash
# Full suite — quiet output
pytest -q

# Single module
pytest -k test_api

# With coverage report
pytest --cov=nv_maser --cov-report=term-missing
```

The CI runs the complete suite across Python 3.10, 3.11, and 3.12.  All tests must pass before a PR can be merged.

---

## Linting

```bash
ruff check src/ tests/ scripts/ --select E,F,W --ignore E501
```

The same command runs in the `lint` CI job. Fix all reported issues before opening a PR. Auto-fix with `ruff check --fix` where safe.

---

## Branch Naming

| Prefix | When to use |
|---|---|
| `feat/short-description` | New functionality |
| `fix/issue-summary` | Bug fixes |
| `docs/what-changed` | Documentation only |
| `perf/what-improved` | Performance improvements |
| `refactor/scope` | Internal restructuring with no behaviour change |
| `test/scope` | Test-only additions or fixes |
| `chore/scope` | Dependency bumps, CI changes, tooling |

Branch names should be lowercase and hyphenated, e.g. `feat/onnx-export` or `fix/lstm-hidden-state-leak`.

---

## Commit Convention

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <one-line summary ≤72 characters>

<optional body — explains WHY, not WHAT>
```

| Type | Use for |
|---|---|
| `feat:` | New capability visible to users |
| `fix:` | Bug correction |
| `perf:` | Measurable performance improvement |
| `docs:` | Documentation changes only |
| `test:` | Adding or correcting tests |
| `refactor:` | Code restructuring, no behaviour change |
| `chore:` | Dependency updates, CI, tooling, build |

**Examples**

```
feat: add ONNX export sub-command

Exporting to ONNX unblocks deployment to edge hardware that cannot
run a full Python runtime.  Tested with ONNXRuntime 1.17 on ARM64.
```

```
fix: extract model_state key from checkpoint dict

Checkpoints written by Trainer wrap weights under a model_state key.
Direct load_state_dict on the raw dict was silently failing on new
checkpoints while succeeding on legacy ones.
```

---

## Pull Request Process

1. **Open a draft PR early** if you want design feedback before implementation is complete.
2. **Fill out the PR template** (`.github/PULL_REQUEST_TEMPLATE.md`) — every section matters.
3. **Request one reviewer** — tag a maintainer or domain expert.
4. **One review approval required** before merge.
5. **Squash-merge** into `main` to keep history linear; ensure the squash commit message follows the convention above.
6. Delete the source branch after merge.

---

## Architecture Decision Records

Write an ADR whenever you:

- Introduce a new module or significant abstraction
- Change a fundamental design paradigm (e.g. switch optimiser, change data format)
- Make a performance trade-off that affects API contracts
- Choose one technology over documented alternatives

**Location:** `docs/adr/`

**Naming:** `ADR-<NNN>-short-title.md` (e.g. `ADR-005-onnx-export-format.md`)

**Template:**

```markdown
# ADR-NNN: Title

## Status
Proposed | Accepted | Superseded by ADR-XXX

## Context
Why does this decision need to be made?

## Decision
What was decided?

## Consequences
What are the trade-offs and follow-on implications?
```

---

## Agent Workflow

This project uses VS Code specialist subagent personas for parallel development sprints. Each agent owns a domain area and works independently within a sprint; results are merged at the end. See [AGENTS.md](AGENTS.md) for the full persona list.

**Conventions:**

- Each agent opens its own feature branch (`feat/<agent>-<sprint>`)
- Agents do not modify files outside their declared scope
- Cross-cutting concerns (config, pyproject.toml) are coordinated by the Architect agent
- Sprint summaries are committed to `docs/session_logs.md` after each sprint

---

## Docker

```bash
# Build the image
docker build -t nv-maser-twin .

# Run the inference server
docker compose up

# Run one-off training inside the container
docker run --rm -v "$(pwd)/checkpoints:/app/checkpoints" nv-maser-twin \
    python -m nv_maser train --arch cnn --epochs 50
```

The `docker-compose.yml` mounts `./checkpoints` as a volume so trained weights persist between container restarts.

---

## Config Overrides

The simulation is fully driven by `config/default.yaml`.  Override any value at runtime:

```bash
python -m nv_maser train --config my_config.yaml
```

`my_config.yaml` is **deep-merged** on top of the defaults — you only need to specify the keys you want to change:

```yaml
# my_config.yaml — override grid resolution and training epochs only
grid:
  nx: 128
  ny: 128
training:
  epochs: 200
```

Add new `SimConfig` fields to both `src/nv_maser/config.py` and `config/default.yaml`, and update `CHANGELOG.md` under `[Unreleased]`.
