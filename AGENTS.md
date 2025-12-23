## Code style

- Do not write meta-comments in source code. Comments and docstrings should describe the actual source code for the reader.
- Never write hot-fixes with use of "if" conditional blocks or "try" statements.
- Do not import libraries or modules inside functions or classes. All imports have to be placed at the top of the file.

## Finalizing request

- Any documentation files should only consist of concrete information and facts. A "waffle" style of writing is not allowed.
- Do not create any summary documents of recent changes. You are only allowed to update existing documentation files. If you built a new component or module, you are allowed to create a new documentation file.
- Summary of done work can be just prompted in chat instead of providing new documentation files.
- Do not create auxiliary versions of implementation (e.g. "v2", "\_refactor"). Work on original implementation and keep clean project's structure.

## Safety

- You are free to use destructive commands as you work in isolated Docker container.
