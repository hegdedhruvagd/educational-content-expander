# AskThili Educational Content Expander

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/5afh9jxjg7/badge.svg)](https://ask.thili.ai/b/5afh9jxjg7)

A small command-line tool that expands educational phrases with a local statistical language model. It uses a bundled, original mini-corpus and bigram/trigram next-word counts with configurable add-k smoothing. Generation does not call an LLM or any external API, and setup does not download a dataset.

## Requirements and setup

Python 3.12 or newer is recommended. The n-gram implementation uses only the Python standard library. Install the project requirements (needed by the preserved starter smoke test):

```bash
python3.12 -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

There is no `.env` setup. No API key or model service is used.

## Usage

Show all commands and options:

```bash
python -m app.cli --help
python -m app.cli expand --help
```

Train the default model from `app/data/educational_corpus.txt` and save it to `data/model.json`:

```bash
python -m app.cli train
python -m app.cli train --output models/classroom.json --k 0.5
```

Expand a prompt directly from the bundled corpus, or load a previously saved model. `--words` controls the number of added words (the prompt is kept as entered):

```bash
python -m app.cli expand "Plants use sunlight" --words 12
python -m app.cli expand "Fractions describe equal parts" --model data/model.json --words 10
```

Example:

```text
$ python -m app.cli expand "Plants use sunlight" --words 8
Plants use sunlight water and carbon dioxide to make food through
```

Use `python -m app.cli train --help` to see training options. `python -m app.cli hello` remains available as a quick plumbing check.

## Architecture

- `app/data/educational_corpus.txt` contains short original lessons in science, math, and reading. Each non-empty line is a document.
- `app/core.py` loads UTF-8 corpus files, lowercases and tokenizes words, counts unigrams, bigrams, and trigrams, calculates add-k conditional probabilities, saves/loads JSON models, and generates continuations.
- `app/cli.py` provides `train`, `expand`, and `hello` commands. If `expand` receives no model path, it trains from the bundled corpus in memory.
- `app/storage.py` and `app/llm.py` are preserved starter modules. The expander does not use the starter chat client.

The model uses up to the last two words as context. It chooses the highest-probability observed next token, using shorter context and corpus frequency to break ties. Add-k smoothing gives nonzero probability to unseen words and contexts when `k > 0`; unknown prompt tokens are handled safely. Output is deterministic and intentionally limited by the small corpus, so it is best suited to short, on-topic lesson starters rather than polished long-form writing.

### Python API

```python
from app.core import generate_text, train_model

model = train_model(k=0.5)
print(generate_text("Plants use sunlight", model=model, max_words=12))
```

## Tests

Run the full test suite from the repository root:

```bash
python -m pytest -q
```

The tests cover corpus loading and preprocessing, bigram/trigram counts and smoothing, prompt continuation and length, model persistence, CLI help/errors, and the original starter smoke tests.

## Milestones

- [x] **1. Corpus and preprocessing**
- [x] **2. Bigram/trigram model with add-k smoothing**
- [x] **3. Educational text generation**
- [x] **4. Command-line interface**
