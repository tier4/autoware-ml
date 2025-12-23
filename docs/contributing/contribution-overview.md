---
icon: lucide/git-pull-request
---

# Contribution Overview

Thank you for your interest in contributing to Autoware-ML! All types and sizes of contribution are welcome.

## Code of Conduct

To ensure the Autoware-ML community stays open and inclusive, please follow the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful and constructive.

## What should I know before I get started?

### Framework Concepts

To gain a high-level understanding of Autoware-ML's architecture and design, see:

- [Framework Design](../framework/design.md)

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** (see [Dev Containers](../getting-started/devcontainer.md))

```bash
git clone https://github.com/YOUR_USERNAME/autoware-ml.git
cd autoware-ml
```

### Code Quality

We use pre-commit hooks to enforce code quality. Run:

```bash
pre-commit run -a --config .pre-commit-config.yaml
```

The hooks check code formatting, linting, and file validity.

## How can I get help?

Open a GitHub Issue for questions, bug reports, or feature requests. Include steps to reproduce for bugs.

## How can I contribute?

### Issues

Participate by:

- Opening issues for questions or feature requests
- Commenting on existing issues
- Helping answer questions from other contributors

### Bug Reports

Before reporting a bug:

1. Search existing issues to see if it's already reported
2. Create an issue with minimal steps to reproduce
3. If you want to fix it, discuss the approach with maintainers first

### Pull Requests

You can submit pull requests for:

- Minor documentation updates
- Bug fixes
- Small feature additions
- Code improvements

For large changes:

1. Open a GitHub Issue to propose the change and discuss the approach
2. Wait for maintainer feedback and consensus
3. Create a pull request referencing the issue
4. Add documentation if relevant

See [Pull Request Guidelines](pull-request-guidelines.md) for detailed instructions on creating PRs, including commit message format, DCO sign-off requirements, and PR checklist.

### Adding Models

See [Adding Models](adding-models.md) for a complete guide on implementing new models.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
