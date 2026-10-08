# Contributing Guide

Thank you for your interest in contributing to Mapping Networks. Contributions are welcome, including bug fixes, new features, documentation improvements, and suggestions.

## Reporting Issues

Before opening an issue:

* Check whether the issue already exists.
* Provide a clear description of the problem.
* Include steps to reproduce the issue.
* Attach logs or screenshots when relevant.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/arjunmnath/marn.git
cd marn
```

### 2. Install dependencies

```bash
make install
```

### 3. Activate the Poetry shell

```bash
make shell
```

## Updating Dependencies

```bash
make update
```

## Running Tests

Run the test suite:

```bash
make test
```

Run tests with coverage:

```bash
make test-cov
```

## Linting and Formatting

Run Ruff linter:

```bash
make lint
```

Format the code:

```bash
make format
```

Run type checking:

```bash
make typecheck
```

Run all checks:

```bash
make check
```

## Building the Package

```bash
make build
```

## Documentation

Build the documentation:

```bash
make docs
```

Serve the generated documentation locally:

```bash
make serve-docs
```

Live rebuild documentation during development:

```bash
make docs-live
```

Clean documentation artifacts:

```bash
make docs-clean
```

## Cleaning Build Artifacts

```bash
make clean
```

## Versioning and Releases

Before creating a release:

```bash
make release
```

Increment the patch version:

```bash
make patch
```

Increment the minor version:

```bash
make minor
```

Increment the major version:

```bash
make major
```

## Branching

Create a dedicated branch for each contribution:

```bash
git checkout -b feature/my-feature
```

Examples:

* `feature/add-layer-normalization`
* `fix/improve-error-handling`
* `docs/update-api-reference`

## Commit Messages

Use descriptive commit messages:

```text
feat: add support for custom mappings
fix: handle empty tensors correctly
docs: improve installation instructions
refactor: simplify encoder implementation
```

## Pull Requests

Before submitting a pull request:

* Run `make format`
* Run `make check`
* Update documentation if needed
* Keep changes focused and self-contained

When opening a PR:

1. Push your branch.
2. Open a pull request.
3. Explain the changes.
4. Reference related issues if applicable.

## Documentation Contributions

Documentation improvements are always welcome:

* Fix typos.
* Improve explanations.
* Add examples.
* Expand API documentation.

## Code Review

Maintainers may request changes before merging. Please be open to feedback and discussion.

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.
