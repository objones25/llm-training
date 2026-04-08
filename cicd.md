# cicd.md

## 🔁 1. CI on Every Push / PR — `ci.yml`

This is the most essential workflow and should run on every push and pull request.

**What it does:**

- Sets up Python and installs dependencies (`pip install -e ".[dev]"`)
- Runs `pytest` for your unit/integration tests
- Checks code formatting with `black` and linting with `ruff` or `flake8`
- Type-checks with `mypy` (especially useful for your FastAPI routes)

CI focuses on automatically testing and integrating code changes into the main branch — for ML projects this means testing model accuracy, performance, and behavior with each update. For your GPT-2 build, this means catching tensor shape bugs, broken tokenizer logic, or broken API endpoints before they land in `main`.

---

## 🧪 2. Model Smoke Tests — `model_tests.yml`

Not the same as unit tests. These validate that your model _actually runs_ correctly.

**What it does:**

- Runs a **fast forward pass** on a tiny dummy input (e.g., `seq_len=16`, `vocab_size=256`) to verify no runtime errors
- Checks that loss goes down for at least N steps on a tiny synthetic dataset (a basic sanity training loop)
- Validates that your checkpoint save/load roundtrip works

This should run on CPU only (no GPU needed for smoke tests) so it works on free GitHub-hosted runners. Keep it under 2–3 minutes.

---

## 🏗️ 3. FastAPI Health & Contract Tests — included in `ci.yml` or separate

Since you're building an inference API around the model:

**What it does:**

- Spins up the FastAPI app with `TestClient` from `httpx`/`starlette`
- Tests all your route contracts (`/generate`, `/health`, `/tokenize`, etc.)
- Validates response schemas (status codes, JSON shape)

This is pure pytest — no GPU needed. Catches broken API contracts the moment you change a route or model interface.

---

## 🐳 4. Docker Build Test — `docker.yml`

LLM applications face unique deployment challenges around GPU drivers, CUDA libraries, and Python environment management that traditional web apps don't encounter. Catching a broken `Dockerfile` early is very valuable.

**What it does:**

- Builds your Docker image (the one that wraps your FastAPI + model)
- Does NOT push to a registry (for a personal project) — just validates it builds successfully
- Optionally runs a `docker run --rm <image> python -c "from model import GPT2; ..."` smoke test inside the container

Trigger: on changes to `Dockerfile`, `requirements.txt`, or `pyproject.toml`.

---

## 📦 5. Dependency & Security Audit — `security.yml`

Simple but high-value, especially since PyTorch pulls in a lot of transitive deps.

**What it does:**

- Runs `pip-audit` or `safety check` to flag known CVEs in your dependencies
- Optionally runs `bandit` for static security analysis of your Python code

Trigger: weekly schedule (`cron`) + on PRs that touch `requirements.txt`.

---

## 📊 6. Training Regression Check (Optional but 🔥 for LLM work) — `training_check.yml`

Language models are best evaluated using structured test datasets rather than isolated test cases — this means reference-based evals comparing responses against expected ones.

For a GPT-2 project specifically, you can do a lightweight version of this:

**What it does:**

- Runs 50–100 training steps on a tiny fixed dataset
- Asserts that `final_loss < initial_loss` and loss is below a threshold (e.g., `< 6.0` for your vocab size)
- Reports the loss curve as a PR comment using [CML](https://cml.dev/)

This catches silent regressions — e.g., a broken attention mask, wrong positional embedding, or learning rate getting zeroed out. Trigger on PRs to `main` only (since it's heavier).

---

## Summary Table

| Workflow                               | Trigger                 | GPU Needed?              | Priority             |
| -------------------------------------- | ----------------------- | ------------------------ | -------------------- |
| `ci.yml` (lint + pytest + API tests)   | Every push/PR           | ❌                       | 🔴 Must have         |
| `model_tests.yml` (forward pass smoke) | Every push/PR           | ❌                       | 🔴 Must have         |
| `docker.yml` (build validation)        | Dockerfile/deps changes | ❌                       | 🟠 High              |
| `security.yml` (dep audit)             | Weekly + deps PRs       | ❌                       | 🟡 Nice to have      |
| `training_check.yml` (loss regression) | PRs to main             | ❌ (CPU ok for tiny run) | 🟠 High for LLM work |

---

**One practical tip:** use `pytest -m "not slow"` markers to separate your fast unit tests (run always) from heavier integration tests (run only on PRs to `main`). This keeps your CI fast during active development.
