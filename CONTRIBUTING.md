# Contributing to Python FCA Libraries Analysis

Thank you for your interest in contributing to this project! We welcome contributions from everyone, whether you're
fixing bugs, adding new features, improving documentation, or sharing your experience with FCA libraries.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Issues](#reporting-issues)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to
uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Prioritize the community's best interests

## How Can I Contribute?

### Reporting Bugs

Before submitting a bug report:

- Check the existing issues to avoid duplicates
- Ensure you're using the latest version
- Test with different FCA libraries if applicable

When submitting a bug report, include:

- Clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- System information (OS, Python version, library versions)
- Code samples or data (when appropriate)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

- Check existing issues and discussions
- Provide clear rationale for the enhancement
- Describe the proposed solution
- Consider backward compatibility
- Include examples of how it would be used

### Documentation Improvements

Documentation contributions are highly valued:

- Fix typos or grammatical errors
- Improve clarity of explanations
- Add examples or use cases
- Update outdated information
- Translate documentation

### Code Contributions

Areas where code contributions are welcome:

- Bug fixes
- Performance improvements
- New library evaluations
- Additional FCA algorithms comparison
- Test coverage improvements
- Example notebooks and tutorials

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/KonchakivskyiDanylo/python-fca-analysis.git
   cd python-fca-analysis
   ```

3. **Create a development environment**:
   ```bash
   python -m venv dev_env
   source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "Add descriptive commit message"
   ```

6. **Push to your fork** and submit a pull request

## Pull Request Process

### Before Submitting

- [ ] Run tests: `python -m pytest`
- [ ] Check code style: `flake8 src/`
- [ ] Update documentation if needed
- [ ] Add tests for new functionality
- [ ] Ensure all checks pass

### Pull Request Guidelines

1. **Title**: Use clear, descriptive titles
    - Good: "Add support for FCA stability metrics"
    - Bad: "Fix stuff"

2. **Description**: Provide detailed information about:
    - What changes were made
    - Why the changes were necessary
    - How to test the changes
    - Any breaking changes

3. **Small PRs**: Keep pull requests focused and reasonably sized

4. **Tests**: Include tests for new features or bug fixes

5. **Documentation**: Update relevant documentation

### Review Process

- All PRs require at least one review
- Maintainers may request changes
- Once approved, maintainers will merge the PR
- We aim to review PRs within 48-72 hours

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://peps.python.org/pep-0008/) with these specifics:

- Line length: 88 characters (Black formatter default)
- Use meaningful variable and function names
- Include docstrings for public functions and classes
- Type hints are encouraged

### Code Formatting

We use the following tools:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Run formatting before committing:

```bash
black src/
isort src/
```

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer(s)]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements

Examples:

- `feat(analysis): add support for concept stability metrics`
- `fix(fcapy): resolve memory leak in large datasets`
- `docs(readme): update installation instructions`

### Documentation Style

- Use clear, concise language
- Include code examples where helpful
- Follow Markdown best practices
- Use proper grammar and spelling

## Reporting Issues

### Bug Reports

Use the bug report template and include:

- **Environment**: OS, Python version, library versions
- **Description**: Clear description of the bug
- **Reproduction**: Minimal code to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Additional context**: Screenshots, logs, etc.

### Feature Requests

Use the feature request template and include:

- **Problem**: What problem does this solve?
- **Solution**: Proposed solution
- **Alternatives**: Alternative solutions considered
- **Additional context**: Any other relevant information

## Development Setup

### Project Structure

```
python-fca-analysis/
├── src/                 # Source code
├── tests/              # Test files
├── docs/               # Documentation
├── examples/           # Example notebooks
├── data/               # Sample datasets
├── requirements.txt    # Production dependencies
├── requirements-dev.txt # Development dependencies
└── setup.py           # Package setup
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_library_comparison.py

# Run with coverage
python -m pytest --cov=src
```

### Library Evaluation Criteria

When evaluating new FCA libraries, consider:

- **Functionality**: Concepts, implications, stability, support
- **Performance**: Speed and memory usage
- **Usability**: API design and documentation
- **Maintenance**: Activity and community support
- **Compatibility**: Python version and dependency requirements

## Questions?

If you have questions that aren't covered in this guide:

- Check existing issues and discussions
- Open a new issue with the "question" label
- Contact the maintainers directly

Thank you for contributing to Python FCA Libraries Analysis!