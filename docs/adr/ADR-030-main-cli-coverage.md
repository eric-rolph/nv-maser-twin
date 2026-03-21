# ADR-030 — CLI Entry-Point Unit Tests (main.py coverage)

**Status**: Accepted  
**Date**: SS26  
**Deciders**: Engineering team

---

## Context

`src/nv_maser/main.py` (125 statements) was at **0 % test coverage**.  The
existing `tests/test_cli.py` file exercises CLI commands via `subprocess.run`,
which produces correct functional behaviour but contributes zero lines to the
Python coverage report because the subprocess runs in a separate process.

`main.py` contains:
- `_deep_merge_config` — a pure merge helper (untested)
- `main()` — argparse routing for 7 sub-commands (untested)
- `cmd_train / cmd_demo / cmd_evaluate / cmd_visualize_coils` — function bodies
- `cmd_dataset / cmd_export / cmd_serve` — function bodies

The 0 % gap was identified during SS26 coverage analysis as the largest single
remaining coverage hole in the non-UI, non-optional-dep modules.

---

## Decision

Write `tests/test_main.py` with **10 test classes, 49 tests**, directly importing
`nv_maser.main` and using `unittest.mock.patch` for all heavy dependencies
(`Trainer`, `FieldEnvironment`, `uvicorn.run`, `build_dataset`, `export_model`,
matplotlib, etc.).

Key design choices:

| Choice | Rationale |
|--------|-----------|
| Direct import, not subprocess | Allows coverage.py to instrument the code |
| Patch at module origin, not at `main` namespace | Local imports inside `cmd_*` functions bind from the original module; patching the origin (e.g. `nv_maser.model.training.Trainer`) captures all binding sites |
| Real torch tensors for `cmd_evaluate` | The inline `torch.einsum` / boolean masking cannot be mocked away without replacing torch itself; using tiny (500×4×4) tensors runs in <10 ms |
| `cmd_demo` deliberately untested | `run_dashboard` opens a live Tkinter/matplotlib GUI; cannot be invoked headlessly without an X server; 7 lines left uncovered |
| `if __name__ == "__main__"` guard left | Standard CPython pattern; coverage ignores it via `pragma: no cover` convention |

### Patch target mapping

| Function | Import inside `cmd_*` | Patch target |
|----------|-----------------------|--------------|
| `cmd_train` — `Trainer` | `from .model.training import Trainer` | `nv_maser.model.training.Trainer` |
| `cmd_train` — tracker | `from .tracking import ExperimentTracker` | `nv_maser.tracking.ExperimentTracker` |
| `cmd_train` — plot | `from .viz.plots import plot_training_history` | `nv_maser.viz.plots.plot_training_history` |
| `cmd_evaluate` — `Trainer` | same | `nv_maser.model.training.Trainer` |
| `cmd_dataset` — `build_dataset` | `from .data.dataset import build_dataset` | `nv_maser.data.dataset.build_dataset` |
| `cmd_export` — `export_model` | `from .export import export_model` | `nv_maser.export.export_model` |
| `cmd_serve` — `uvicorn` | `import uvicorn` | `uvicorn.run` |
| `cmd_visualize_coils` — env | `from .physics.environment import FieldEnvironment` | `nv_maser.physics.environment.FieldEnvironment` |
| `cmd_visualize_coils` — plots | `from .viz.plots import plot_coil_influence, ...` | `nv_maser.viz.plots.plot_coil_influence` etc. |
| `cmd_visualize_coils` — plt | `import matplotlib.pyplot as plt` | `matplotlib.pyplot.show` |

---

## Consequences

### Positive
- `main.py` coverage: **0 % → 94 %** (118/125 statements reached)
- `_deep_merge_config` has exhaustive pure-function tests (8 cases, including
  immutability and nested partial override preservation)
- All 7 sub-command dispatch paths verified
- CLI flag overrides (`--device`, `--arch`, `--epochs`, `--samples`) verified
  to correctly mutate `SimConfig`
- YAML config loading path verified, including `ValidationError → SystemExit(1)`
- Full suite: **2523 passed, 75 skipped** (baseline 2474 + 49 new)

### Neutral
- `tests/test_cli.py` (subprocess-based) remains as integration tests; they
  are complementary — subprocess tests verify real binary behaviour while the
  new unit tests provide coverage and fast feedback.

### Negative / Limitations
- `cmd_demo` (lines 47–53) remains uncovered; would require a headless display
  server or deep mocking of the dashboard's event loop.
- The `if __name__ == "__main__"` guard (line 226) is structurally
  untestable without running the module as a script.

---

## Covered Lines Summary

```
src/nv_maser/main.py    125 stmts   7 miss   94 %   Missing: 47-53, 226
```

Uncovered lines:
- **47–53** — `cmd_demo` body (`run_dashboard` call, GUI)
- **226** — `if __name__ == "__main__": main()`
