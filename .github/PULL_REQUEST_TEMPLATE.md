## Description

<!-- What problem does this solve? Link to an issue if applicable (e.g. Closes #42) -->

## Type of change

- [ ] Bug fix (non-breaking fix for an existing issue)
- [ ] New feature (non-breaking additive change)
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactor (no functional change)
- [ ] Tests only

## Checklist

- [ ] Tests pass locally: `pytest -q`
- [ ] Linting is clean: `ruff check src/ tests/ scripts/ --select E,F,W --ignore E501`
- [ ] New code has tests — or explain why not: <!-- your explanation here -->
- [ ] ADR created or updated in `docs/adr/` if an architectural decision was made
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] `config/default.yaml` updated if new `SimConfig` keys were added
- [ ] Docker image still builds: `docker build .`

## Notes for reviewer

<!-- Anything the reviewer should pay particular attention to, or that guided design decisions -->
