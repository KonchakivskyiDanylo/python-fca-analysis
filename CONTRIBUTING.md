# 🤝 Contributing to Python FCA Libraries Analysis

Thanks for your interest in improving this project! Whether you're fixing a bug, writing a new feature, improving docs,
or analyzing rules — we appreciate your time and ideas.

This project is part of a broader research initiative, and your contributions can help both academic and open-source
communities.

## 📚 Table of Contents

- [🧭 Code of Conduct](#-code-of-conduct)
- [🌱 First-Time Contributors](#-first-time-contributors)
- [🚀 Ways to Contribute](#-ways-to-contribute)
- [🛠 Getting Started](#-getting-started)
- [📥 Pull Request Process](#-pull-request-process)
- [✨ Code Style Guidelines](#-code-style-guidelines)
- [🐛 Issue Reporting](#-issue-reporting)
- [🧠 Evaluation Criteria (for new FCA libraries)](#-evaluation-criteria-for-new-fca-libraries)

## 🧭 Code of Conduct

We are committed to maintaining a welcoming and inclusive space. By participating, you agree to follow our Code of
Conduct.

### Our Standards

- Be respectful and constructive
- Assume good intentions
- Keep discussions focused and clear
- Acknowledge mistakes and learn
- Encourage collaboration

## 🌱 First-Time Contributors

We’re beginner-friendly!  
If you’re unsure how to start, check out:

-

Open [issues labeled "good first issue"](https://github.com/KonchakivskyiDanylo/python-fca-analysis/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

- Or just open an issue/question — we’re happy to guide you

You can also contribute to:

- Documentation & typos
- Test coverage
- Code cleanup & formatting

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

## 🚀 Ways to Contribute

### 🐞 Report Bugs

- Describe what happened and what you expected
- Share code/data to reproduce
- Mention OS, Python version, and installed libraries

### 💡 Suggest Enhancements

- What problem would the change solve?
- How would the feature be used?
- Is it compatible with current design?

### 🛠 Submit Code

- Add new functionality
- Optimize performance or usability
- Extend to other FCA libraries or datasets

## 🛠 Getting Started

1. **Fork and clone the repo**
   ```bash
   git clone https://github.com/your-username/python-fca-analysis.git
   cd python-fca-analysis
   ```

2. **Set up your environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a new branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## 📥 Pull Request Process

Before submitting:

* ✅ Run all tests (`pytest`)
* ✅ Add tests for new logic
* ✅ Update docs or `README.md` if needed

Pull request format:

```
feat(utils): add support for visualizing concept lattice

Adds a new function `visualize_lattice()` that plots FCA lattice from the context matrix.
```

## ✨ Code Style Guidelines

### Python

* Follow [PEP 8](https://peps.python.org/pep-0008/)

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

* `feat`: New feature
* `fix`: Bug fix
* `docs`: Only docs
* `test`: Add or fix tests
* `refactor`: Code change that doesn’t affect behavior

Examples:

* `fix(rules): correct confidence threshold in graph plot`
* `docs(readme): add quick demo section`
* `feat(data): add new preprocessing step for quantile binning`

## 🐛 Issue Reporting

### Bug Report

Please include:

* OS, Python version, relevant package versions
* Minimal reproducible example
* Expected vs actual behavior
* Screenshots or logs if helpful

### Feature Request

Please describe:

* What problem this solves
* How it could be implemented
* Any alternatives considered

## 🧠 Evaluation Criteria (for new FCA libraries)

When adding new libraries, please evaluate:

* ✅ Concept generation
* ✅ Implication support
* ✅ Performance on medium/large datasets
* ✅ Usability (API, docs, install)
* ✅ Community activity

## 🙌 Thank You!

Whether it's your first contribution or you’re a regular — thank you for helping make this project better for everyone.
