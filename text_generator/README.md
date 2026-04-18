# text-generator

A lightweight text generation project using configurable YAML settings, Hugging Face models, and Poetry for dependency management.

---

## 📦 Requirements

- **Python** 3.10+
- **Poetry** (dependency and virtual environment manager)
  - Install via `pipx install poetry`
  - Or: `pip install --user poetry`
- **pre-commit** (Git hooks, installed inside the project)

> 💡 Tip: If you use VS Code or PyCharm, select Poetry’s virtual environment as your interpreter:
> `poetry env info --path`

---

## 📂 Repository Structure

```
text-generator/
├─ config/
│  └─ news-simple.yaml
├─ src/
│  └─ text_generator/
│     ├─ __init__.py
│     ├─ main.py
│     ├─ generator.py
│     ├─ config_loader.py
│     └─ ...
├─ pyproject.toml             # metadata + dependencies (Poetry)
└─ .pre-commit-config.yaml    # Git hooks
```

---

## ⚙️ Installation

```bash
# 1) Install the project (creates/uses Poetry venv)
poetry install

# 2) (Optional) Enter Poetry shell
poetry shell
```

If you don’t activate the shell, prefix commands with `poetry run`, e.g.:

```bash
poetry run python src/text_generator/main.py
```

---

## 📝 Configuration (YAML)

Example: `config/news-simple.yaml`

```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  device: "auto"

decoding:
  temperature: 0.7
  top_p: 0.92
  top_k: 50
  repetition_penalty: 1.05
  max_new_tokens: 200
  do_sample: true
  num_return_sequences: 1
  num_beams: 1


news:
  length_min: 80
  length_max: 140
```

Run directly:

```bash
poetry run python src/text_generator/main.py
```

> ⚠️ If you see `ModuleNotFoundError: transformers`, make sure you always run with `poetry run ...` or inside `poetry shell`.

---

## 📚 Dependency Management (Poetry)

- Add dependency:
  ```bash
  poetry add transformers
  poetry add "numpy==1.26.4"        # exact version
  poetry add "pydantic>=2.7,<3"     # version range
  ```

- Dev dependencies (linters, tests, formatting):
  ```bash
  poetry add -D pre-commit black ruff pytest
  ```

- Update all:
  ```bash
  poetry update
  ```

- Inspect environment:
  ```bash
  poetry env info
  poetry show
  ```

---

## ✅ Pre-commit (formatting, linting, security)

Minimal config: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
        args: ["--line-length=88"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.7
    hooks:
      - id: ruff
        args: ["--fix"]
```

Install & test hooks:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

Add a custom pytest hook (prototype):

```yaml
repo: local
hooks:
  - id: pytest
    name: Run pytest
    entry: poetry run pytest
    language: system
    pass_filenames: false
```

Then refresh:

```bash
pre-commit install --overwrite
pre-commit autoupdate
```

---

## 🔍 Troubleshooting

- **`ModuleNotFoundError` after install**
  → Always run with `poetry run ...` or inside `poetry shell`.

- **`Permission denied: 'src/.../main.py'`**
  → Don’t execute the file directly. Use:
  `poetry run python src/text_generator/main.py`

- **pre-commit says “no files to check”**
  → Ensure the file is tracked with `git add` and not ignored.

- **`git push` requires a remote**
  ```bash
  git remote add origin <URL>
  git push -u origin main
  ```

---

## 🚀 Recommended Workflow (from scratch)

```bash
# Clone / create repo
git clone <url> text-generator && cd text-generator

# Install dependencies
poetry install

# Setup pre-commit
poetry run pre-commit install
poetry run pre-commit run --all-files

# Commit & push
git add -A
git commit -m "feat: initial text generator pipeline"
git push -u origin main
```

---
