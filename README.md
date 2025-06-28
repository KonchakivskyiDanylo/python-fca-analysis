# Python FCA Libraries Analysis for Sociological Research

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project analyzes and compares Python libraries for Formal Concept Analysis (FCA) with a focus on sociological data.
It was developed as part of a collaborative research initiative (KSE Graph Theory Lab) and applies FCA to
multi-year survey datasets to extract patterns, rules, and conceptual structures.

## Table of Contents

- [Overview](#overview)
- [What is FCA?](#what-is-fca)
- [Installation](#installation)
- [Quick Demo](#quick-demo)
- [Library Comparison](#library-comparison)
- [Use Cases](#use-cases)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🧠 Overview

Formal Concept Analysis (FCA) helps discover hierarchical relationships between objects and attributes in binary data.
This project benchmarks three Python FCA libraries (`fcapy`, `concepts`, `fca-algorithms`) using real sociological
survey datasets (ESS rounds 1–9).

Goals:

- Benchmark FCA libraries on usability, performance, and feature set.
- Extract association rules and concept lattices from survey data.
- Analyze trends across years and track rule stability.

## What is Formal Concept Analysis?

Formal Concept Analysis transforms tabular data into **concept lattices**, revealing relationships between attributes.

### Key Terms:

- **Formal Context**: Binary relation between objects and attributes.
- **Formal Concept**: Pair of objects and their shared attributes.
- **Lattice**: A hierarchy of formal concepts.
- **Implications**: "If A then B" logic rules.

➡️ Learn more:

- [Wikipedia](https://en.wikipedia.org/wiki/Formal_concept_analysis)
- [Intro Video](https://www.youtube.com/watch?v=fJu_bV9MKfM)

## ⚙️ Installation

```bash
git clone https://github.com/KonchakivskyiDanylo/python-fca-analysis.git
cd python-fca-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Library Comparison

| Library            | Concepts | Implications | Stability | Support | Ease of Use | Performance | Documentation |
|--------------------|----------|--------------|-----------|---------|-------------|-------------|---------------|
| **fcapy**          | ✅ Full   | ❌ No         | ✅ Yes     | ✅ Yes   | ⭐⭐⭐⭐⭐       | ⭐⭐⭐⭐        | ⭐⭐⭐⭐⭐         |
| **concepts**       | ✅ Basic  | ✅ Limited    | ❌ No      | ❌ No    | ⭐⭐⭐         | ⭐⭐          | ⭐⭐⭐⭐          |
| **fca-algorithms** | ✅ Full   | ❌ No         | ❌ No      | ❌ No    | ⭐⭐⭐         | ⭐⭐⭐⭐        | ⭐⭐            |

### Our Recommendation:

* Use `fcapy` for **concept extraction**.
* Use `fca-algorithms` for **implication mining**.

## 📚 Use Cases

* 🧠 **Clustering public opinion** by shared attribute sets
* 📊 **Trend comparison** across ESS survey rounds
* 🛠️ **Survey validation** through rule inconsistency detection
* 🔄 **Demographic cross-tabulation** with FCA-based association rules

## Contributing

## 🤝 Contributing

We welcome contributions of all kinds — from documentation fixes to new algorithms.
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, style guides, and PR process.

## 🪪 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.