# Development and compatibility

## Supported baseline

- Python 3.12
- PyTorch `>=2.6,<3.0`
- CPU execution; CUDA behavior will be covered alongside trainer/device support
- Ordinary `torch.nn.Module` targets without tied parameters or registered parametrizations

## Checks

Run the current test and style suite from the repository root:

```bash
PYTHONPATH=src python -m pytest -q
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy src
```

The repository's existing virtual environment may predate the PyTorch runtime dependency. Run
`poetry install` after pulling dependency changes so tests, type checking, and coverage use one
consistent environment.

## Documentation policy

Documentation is developed with each implementation milestone:

1. Add or update a Markdown page under `docs/` for every new public contract.
2. Use examples that run against the APIs implemented in that milestone.
3. Label planned behavior explicitly instead of documenting it as available.
4. Record unsupported semantics and their failure behavior.
5. Link new pages from `docs/index.md` and update `progress.md` in the same change.

## Design invariants

- Generated tensors keep fully qualified target parameter names.
- Target parameters and persistent buffers are not mutated by mapped forwards.
- Gradients flow from target outputs into generated tensors and, later, latent variables.
- Frozen projection tensors will be buffers, not optimizer-visible parameters.
- Flat vectors remain an implementation detail and do not cross subsystem boundaries.
- Unsupported model semantics fail explicitly rather than being approximated silently.
