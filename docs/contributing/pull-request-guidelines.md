---
icon: lucide/git-merge
---

# Pull Request Guidelines

This page covers how to create effective pull requests for Autoware-ML.

## Before You Start

1. **Check existing PRs** - Someone might already be working on it
2. **Open an issue first** - For significant changes, discuss the approach
3. **Keep PRs focused** - One feature or fix per PR

## Creating a Pull Request

### 1. Branch Naming

Use descriptive branch names:

```text
feat/add-bevfusion-model
fix/lidar-projection-crash
docs/improve-quickstart
refactor/simplify-transforms
```

### 2. PR Title and Commit Messages

Write a clear, concise title and commit message:

```text
feat: add BEVFusion model implementation
fix: handle empty point clouds in projection
docs: add GPU troubleshooting to installation guide
```

Use conventional commit prefixes:

| Prefix      | Use Case                 |
| ----------- | ------------------------ |
| `feat:`     | New feature              |
| `fix:`      | Bug fix                  |
| `docs:`     | Documentation only       |
| `refactor:` | Code restructuring       |
| `test:`     | Adding or updating tests |
| `chore:`    | Maintenance tasks        |

### 3. Signed Commits

We require DCO sign-off for all commits. Use the `-s` flag: `git commit -sm "feat: add my feature"`.

### 4. PR Description

Fill out the PR template completely:

```markdown
## Summary

Brief description of what this PR does.

## Related Issues

Fixes #123
Related to #456

## Checklist

- [x] You've checked our [contribution overview](https://tier4.github.io/autoware-ml/main/contributing/contribution-overview/).
- [x] Your PR follows our [pull request guidelines](https://tier4.github.io/autoware-ml/main/contributing/pull-request-guidelines/).

## Additional Notes

Any context reviewers should know.

## Additional Log and Media

Training logs, performance metrics, images, videos, etc.
```

## PR Checklist

Before marking your PR as ready:

- [ ] **Pre-commit passes**: `Ctrl+Shift+P` -> `Tasks: Run Task` -> `Pre-commit: Run` in VS Code
- [ ] **Tests pass**: `Ctrl+Shift+P` -> `Tasks: Run Task` -> `Python: Test` in VS Code
- [ ] **CSpell passes**: `Ctrl+Shift+P` -> `Tasks: Run Task` -> `CSpell: Check` in VS Code
- [ ] **No new warnings**: Check CI output
- [ ] **Documentation updated**: If you changed behavior
- [ ] **Commit history clean**: Keep Conventional Commits format

## Merging

### Merge Strategy

We use **squash and merge** to keep history clean. Your commits will be combined into one, therefore the PR title should be a concise description of the changes.

### Who Merges?

Maintainers merge approved PRs.

## Special Cases

### Draft PRs

Use draft PRs for:

- Work in progress (WIP)
- Early feedback requests
- CI testing before full review

Convert to ready when complete.

### Large Changes

For significant features:

1. Open an issue
2. Get buy-in on the approach
3. Consider breaking into smaller PRs

### Breaking Changes

If your change breaks backward compatibility:

1. Clearly mark in PR title: `feat!: change config format`
2. Update relevant documentation
3. Notify in PR description

## Getting Help

Stuck on something?

- Ask in the PR comments
- Open an issue
- Tag a maintainer (sparingly)

## Examples of Good PRs

A good PR:

- Has a clear title and description
- Is focused on one change
- Includes tests
- Updates documentation
- Passes CI
- Responds to feedback promptly

## Quick Reference

```bash
# Add your remote
git remote add my-remote https://github.com/my-username/autoware-ml.git

# Start a new feature
git checkout main
git pull
git checkout -b feature/my-feature

# Make changes and commit (with DCO sign-off)
git add .
git commit -sm "feat: add my feature"

# Push and create PR
git push -u my-remote feature/my-feature
# Open PR on GitHub

# Update after review
git add .
git commit -sm "fix: correct array size"
git push
```
