# Contributing to Tensorus

Thank you for your interest in contributing to Tensorus! We welcome contributions from everyone, whether it's reporting a bug, proposing a feature, or submitting a pull request.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Submitting Changes](#submitting-changes)
6. [Code Review](#code-review)
7. [Community Guidelines](#community-guidelines)

## Getting Started

Before you begin, please make sure you have:

1. A GitHub account
2. Git installed on your local machine
3. Python 3.8 or higher
4. Installed the development dependencies

## Development Environment

To set up your development environment:

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/tensorus.git
   cd tensorus
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We follow these guidelines for code style:

1. Follow PEP 8 for Python code
2. Use type hints for all function definitions
3. Document all functions, classes, and modules with docstrings (Google style)
4. Keep line length to a maximum of 100 characters

We use the following tools to enforce code style:
- Black for code formatting
- isort for import organization
- mypy for type checking
- flake8 for linting

Before submitting a pull request, run these tools:

```bash
black .
isort .
mypy .
flake8 .
```

## Testing

All changes should include appropriate tests. We use the standard `unittest` framework:

1. Add tests that cover your changes
2. Make sure all tests pass:
   ```bash
   python run_tests.py
   ```

3. Aim for increased test coverage

## Submitting Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with a descriptive message:
   ```bash
   git commit -m "Add new feature: your feature description"
   ```

3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Submit a pull request from your branch to the main repository

## Code Review

All submissions require review. We use GitHub pull requests for this purpose:

1. Ensure your PR has a clear description of changes
2. Link to any relevant issues
3. Respond to review comments promptly
4. Make requested changes as needed

## Community Guidelines

Please follow these guidelines when participating:

1. Be respectful and inclusive
2. Focus on the technical discussion
3. Provide constructive feedback
4. Help others learn and grow

## License

By contributing to Tensorus, you agree that your contributions will be licensed under the project's MIT License. 