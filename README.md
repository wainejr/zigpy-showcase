# README

Showcase of joining Zig and Python using Cython.

```bash
# Install poetry for venv and package management
# (you can do this without poetry, I just find it easier to use it)
pip install -U poetry

# Build dependencies in Zig and Cython
poetry build
# Install in poetry virtual environment
poetry install

# Run command with example of case
poetry run python zigpy/proto_zig.py
```
