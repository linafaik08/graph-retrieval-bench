# Graph RAG or Vector RAG? NaiveRAG, LightRAG and Microsoft GraphRAG on WildGraphBench
**A Hands-On Benchmark of Retrieval Quality and Cost on Real-World Web Sources**

- Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
- Creation date: September 2026
- Last update: September 2026

## Objective

This repository investigates a practical question in retrieval-augmented generation: **when does a knowledge graph retrieve better answers than plain vector search, and is it worth what it costs to build?**

Vector RAG splits documents into chunks, embeds them and returns the chunks closest to the question. Graph-based RAG goes further: an LLM reads every chunk, extracts entities and relations, and retrieval follows that structure. In principle, this helps with questions whose answer is spread across several documents. In practice, it multiplies the indexing cost.

The notebooks measure both sides on the same questions and documents. Four systems are compared: **NaiveRAG** (the vector baseline), **LightRAG** (hybrid mode) and **Microsoft GraphRAG** (local and global search). Each is scored **per question type**, because the WildGraphBench paper found that no single method wins everywhere. Each is also costed **per phase**: indexing once, then querying for every question.

The final notebook answers the question that matters in practice: **which question types justify a graph index, and at what price per question?**

## Project Description

Most GraphRAG benchmarks use short, clean passages. [WildGraphBench](https://arxiv.org/abs/2602.02053) uses the web pages that Wikipedia articles cite instead: long, noisy, heterogeneous documents full of menus and banners. Each question is built from a Wikipedia statement, and the evidence lives in the pages it cites. This makes the benchmark close to what a retrieval system meets in production.

Three kinds of questions test three abilities:

- **Single-fact**: the answer sits in one page. It tests precise retrieval.
- **Multi-fact**: the answer combines facts from at least two pages. It tests aggregation.
- **Summary**: the answer should cover a whole Wikipedia section. It is graded statement by statement for recall and precision.

### The Four Systems

1. **NaiveRAG** (no framework): 1,200-token chunks, OpenAI embeddings, cosine top-k, one answer call. It is written in a few lines in the notebook, so the baseline hides nothing.
2. **LightRAG** (HKU): an LLM extracts entities and relations from every chunk. At query time, low-level keywords select entities and high-level keywords select relations; `hybrid` mode combines both.
3. **GraphRAG local search** (Microsoft): starts from the entities closest to the question and gathers their relations, source chunks and community reports.
4. **GraphRAG global search** (Microsoft): groups entities into Leiden communities, has the LLM summarise each one, then answers by map-reduce over those summaries.

All four share the same fairness rules, set once in `src/config.py`:
- **Models.** Every system uses `gpt-5.6-luna` for indexing and answers, and `text-embedding-3-small` for embeddings.
- **Chunking.** Chunks are 1,200 tokens with a 100-token overlap.
- **Retrieval budget.** Each question type gets the same retrieval budget.
- **Answer format.** Every system is asked for the same answer format.

Each framework keeps its own answer prompt, so the comparison measures each framework as shipped. Every answer is graded by `gpt-5-mini` with the prompts of the official WildGraphBench evaluator.

### Data

The dataset is [WildGraphBench](https://huggingface.co/datasets/Bstwpy/WildGraphBench) on Hugging Face (Apache-2.0), pinned to one revision. The default domain is **Technology**:
- **Questions.** It has 113 questions: 56 single-fact, 33 multi-fact and 24 summary.
- **Corpus.** The corpus is 441 web pages totalling 3.35 million tokens, cited by the Wikipedia article on Steam.

`src/data.py` downloads the domain on first run into `data/raw/`. The domain is a parameter (`DOMAIN` in `src/config.py`), and any of the twelve WildGraphBench domains works.

**One file is deliberately excluded.** The corpus folder also contains the Wikipedia article itself, which is the text the gold answers were written from. Indexing it would give every system the answer key.

**Every run starts with a subset.** `RUN_MODE = RunMode.SUBSET` keeps 3 questions per type and only the pages they cite (23 pages, 167,000 tokens). This checks the whole pipeline for a few cents and prints a projected cost for the full index. Subset scores are not meaningful, because the subset corpus contains almost only relevant pages. `RunMode.FULL` runs the actual benchmark.

### Code Structure

```
data/                                   # created on first run; the raw dataset is cached here, not tracked
outputs/<domain>/<run_mode>/            # inputs, indexes, predictions, judgments and the usage ledger, not tracked
results/<domain>/full/                  # results.csv and run_manifest.json of the full run

graphrag_project/
├── settings.yaml                       # Microsoft GraphRAG settings (models and folders overridden per run)
└── prompts/                            # GraphRAG indexing and search prompts, as generated by `graphrag init`

notebooks/
├── 01_data_preparation.ipynb           # Download, question types, corpus, subset selection
├── 02_naive_rag.ipynb                  # Chunk, embed, retrieve, generate: vector RAG from scratch
├── 03_lightrag.ipynb                   # Build and inspect a LightRAG graph, answer in hybrid mode
├── 04_graphrag.ipynb                   # Build and inspect a GraphRAG index, answer with local and global search
└── 05_evaluation_and_results.ipynb     # LLM judge, results table, quality and cost figures

papers/                                 # WildGraphBench, LightRAG and GraphRAG papers (PDF)

src/
├── config.py                           # Every parameter of the benchmark, with its link to the papers
├── data.py                             # Download, normalisation, evidence mapping, subset selection, chunking
├── usage_tracking.py                   # Token and cost ledger, OpenAI helpers, LightRAG and LiteLLM adapters
├── question_runner.py                  # Resumable loop that answers every question with one system
├── graphrag_utils.py                   # GraphRAG configuration per run and table loading inside Jupyter
├── evaluation.py                       # WildGraphBench judge, Wilson intervals, results table, run manifest
└── visualization.py                    # Plotly figures for quality and cost
```

The framework calls stay in the notebooks rather than being wrapped in `src/`, so that every `ainsert`, `aquery`, `build_index`, `local_search` and `global_search` call is visible where it is explained.

## How to Use This Repository?

### Requirements

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

Main libraries:
```
# Retrieval frameworks, pinned because their APIs change between releases
lightrag-hku==1.5.7
graphrag==3.1.2

# LLM access and token counting
openai
litellm                # GraphRAG calls models through LiteLLM; its callbacks feed the cost ledger
tiktoken

# Data
huggingface_hub
numpy
pandas
pyarrow
networkx

# Plotting and notebooks
plotly
jupyter
ipykernel
```

Two constraints are worth knowing before installing:

- **Python 3.11 or 3.12.** `graphrag==3.1.2` requires Python 3.11 or newer, and the pinned dependency set was resolved on 3.12.
- **On Apple Silicon, the interpreter must be arm64.** GraphRAG depends on `lancedb`, which ships no Intel-Mac wheel. An x86_64 Python therefore fails dependency resolution. `uv python install cpython-3.12-macos-aarch64-none` provides a suitable interpreter.

### Installation

1. **Install uv** (if not already installed)

2. **Clone the repository**:
```bash
git clone <repository-url>
cd graph-retrieval-bench
```

3. **Install dependencies with uv**:
```bash
# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install the package
uv pip install -e .
```

### Setup

An OpenAI API key is required for indexing, answering and judging:

```bash
cp .env.example .env   # then set OPENAI_API_KEY in .env
```

Every notebook and `graphrag_project/settings.yaml` read the key from this file.

**Cost.** The full Technology corpus is 3.35 million tokens, which is about 3,200 chunks. Both graph frameworks send every chunk through several LLM prompts, so their indexing cost is far higher than NaiveRAG's embedding-only cost. The subset run measures the real cost per token, and notebooks 03 and 04 print a projected full-corpus cost before the full run is launched. Prices are set in `src/config.py` (checked on 2026-09-17), and every cost is recomputed from token counts.

### Running the Project

1. **Start with the subset.** `RUN_MODE` is `RunMode.SUBSET` by default in `src/config.py`.

2. **Run the notebooks in order**, each one top to bottom:
   - `01` downloads the domain, explains the question types and writes the run inputs.
   - `02` builds NaiveRAG step by step and answers every question.
   - `03` builds the LightRAG graph, inspects it and answers in hybrid mode.
   - `04` builds the GraphRAG index, inspects its communities and answers with local and global search.
   - `05` grades every answer and draws the results.

3. **Check the projections**, then switch to `RunMode.FULL` and run the five notebooks again. The subset and full runs are stored in separate folders.

4. **Rerun safely.** Indexes are reused when present, predictions and judgments are cached, and an interrupted cell resumes where it stopped. Deleting an index folder under `outputs/` forces a rebuild of that index.

### Key Features Demonstrated

- **Three frameworks, one set of rules**: models, chunking, retrieval budget and answer format come from a single configuration file, with each value compared to the papers.
- **Vector RAG with no framework**: chunking, embedding, cosine retrieval and generation fit in a few visible cells.
- **Graphs opened up**: the LightRAG graph and the GraphRAG entities, communities and reports are inspected, not just used.
- **Every token accounted for**: one ledger records each LLM and embedding call, per system, phase and question. It collects them through direct OpenAI calls, LightRAG model functions and LiteLLM callbacks for GraphRAG.
- **Cost measured before it is paid**: a subset run projects the full indexing cost of each graph framework.
- **The official grading protocol**: judge prompts are copied from the WildGraphBench evaluator. A documented fix restores summary precision, which the original script always scores as zero.
- **Honest uncertainty**: accuracies come with 95% Wilson intervals, because a 24-question category moves four points per question.
- **No answer key in the corpus**: the Wikipedia article behind the gold answers is excluded from indexing.
- **Reproducible runs**: the dataset revision, library versions, models and prices are recorded in `run_manifest.json` next to the results.

## Resources

- **WildGraphBench Paper**: https://arxiv.org/abs/2602.02053
- **WildGraphBench Code**: https://github.com/BstWPY/WildGraphBench
- **WildGraphBench Dataset**: https://huggingface.co/datasets/Bstwpy/WildGraphBench
- **LightRAG Paper**: https://arxiv.org/abs/2410.05779
- **LightRAG**: https://github.com/HKUDS/LightRAG
- **GraphRAG Paper (From Local to Global)**: https://arxiv.org/abs/2404.16130
- **Microsoft GraphRAG**: https://microsoft.github.io/graphrag/
- **uv Package Manager**: https://github.com/astral-sh/uv

## License

MIT License - Free to use for learning and production projects. The WildGraphBench data and the judge prompts adapted in `src/evaluation.py` are distributed under Apache-2.0.

---

*Built with ❤️ to find out when a knowledge graph is worth its indexing bill*
