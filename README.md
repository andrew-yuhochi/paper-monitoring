# Paper Monitoring

Automated weekly digest of important ML/DS/DL/MLOps research papers, classified by impact tier and linked to foundational knowledge.

> Built with [Claude Code](https://claude.ai/code)

## Setup

### 1. Install Ollama and pull the model

```bash
brew install ollama
ollama pull qwen2.5:7b
```

### 2. Create the Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env if you want to override any PM_ settings (optional)
```

### 4. Seed the knowledge bank (one-time)

```bash
python -m src.seed
```

This fetches landmark and survey papers from arXiv, extracts concepts via Ollama, and
builds the foundational graph. Expect it to take 20-40 minutes on first run.

To add a single paper manually later:

```bash
python -m src.seed --arxiv-id 1706.03762
```

### 5. Run the pipeline manually (optional smoke test)

```bash
python -m src.pipeline
```

Check `digests/YYYY-MM-DD.html` for the output.

---

## Scheduling (Weekly Cron — macOS)

The pipeline is designed to run automatically every Friday at 6 PM via cron.

### Step 1: Grant cron Full Disk Access

macOS Catalina and later silently block cron from writing files unless it has Full Disk
Access. Without this, the pipeline will run but produce no output and log no errors.

1. Open **System Settings** > **Privacy & Security** > **Full Disk Access**
2. Click the **+** button
3. Press **Cmd+Shift+G**, enter `/usr/sbin/cron`, and click **Open**
4. Ensure the toggle next to `cron` is enabled

**Verify it works before relying on the schedule** (see Step 3 below).

### Step 2: Add the crontab entry

Open your crontab:

```bash
crontab -e
```

Add this line, replacing `/path/to` with the absolute path to your clone:

```
0 18 * * 5 caffeinate -i /path/to/paper-monitoring/run_weekly.sh >> /path/to/paper-monitoring/data/logs/cron.log 2>&1
```

Example (adjust to your machine):

```
0 18 * * 5 caffeinate -i /Users/yourname/projects/paper-monitoring/run_weekly.sh >> /Users/yourname/projects/paper-monitoring/data/logs/cron.log 2>&1
```

**About `caffeinate -i`:** This prevents the machine from sleeping due to inactivity
while the pipeline is running. It does **not** prevent sleep when the lid is closed.
Make sure the machine is open and awake on Friday at 6 PM.

### Step 3: Verify cron can write to the project directory

Before waiting for Friday, test that cron has the permissions it needs:

```bash
crontab -e
```

Add a temporary test entry that runs 2 minutes from now:

```
# Replace HH:MM with 2 minutes from now; remove this line after confirming
MM HH * * * touch /path/to/paper-monitoring/data/cron_test.txt
```

Wait 2 minutes, then check:

```bash
ls /path/to/paper-monitoring/data/cron_test.txt
```

If the file exists, cron has write access. Remove the test entry from your crontab
and delete the test file. If the file does not appear, revisit Step 1 — Full Disk
Access is almost always the cause.

---

## Cloning to a second machine

The project is fully self-contained. To run it on a different Mac:

1. Clone the repo
2. Repeat **Setup** steps 1–5 on the new machine
3. Add the crontab entry (Step 2 above) with paths for the new machine
4. Grant Full Disk Access to cron on the new machine (Step 1 above)

`run_weekly.sh` uses paths relative to the script itself, so no edits to the script
are needed — only the crontab entry requires a machine-specific absolute path.

---

## Project structure

```
paper-monitoring/
├── src/
│   ├── pipeline.py          # Weekly pipeline entry point
│   ├── seed.py              # Seeding CLI entry point
│   ├── config.py            # All settings (PM_ env prefix)
│   ├── integrations/        # arXiv, HuggingFace, Ollama, PDF clients
│   ├── models/              # Pydantic data models
│   ├── services/            # PreFilter, OllamaClassifier, ConceptLinker, DigestRenderer, Seeder
│   ├── store/               # GraphStore (SQLite)
│   ├── templates/           # Jinja2 HTML templates
│   └── utils/               # Logging config, normalisation helpers
├── tests/
│   ├── unit/                # Unit tests (no live APIs, no live Ollama)
│   └── integration/         # Integration tests (marked @pytest.mark.slow)
├── digests/                 # Weekly HTML digest output
├── data/
│   ├── logs/                # pipeline.log, cron.log
│   └── paper_monitoring.db  # SQLite graph database
├── run_weekly.sh            # Cron launcher script
├── requirements.txt
└── .env.example
```

## Running tests

```bash
source .venv/bin/activate
pytest tests/unit/           # Fast — no external dependencies
pytest tests/ -m slow        # Includes live arXiv API call
```
