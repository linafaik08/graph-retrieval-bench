# Graph vs vector retrieval benchmark on WildGraphBench

## Goal

We are building a teaching repository that compares four retrieval systems on one domain of
WildGraphBench. The domain is a configuration parameter, and it defaults to `technology`
(113 questions). The four systems are:
- **NaiveRAG**, a plain vector-search baseline.
- **LightRAG** in hybrid mode.
- **Microsoft GraphRAG** in local search mode.
- **Microsoft GraphRAG** in global search mode.

The benchmark answers two questions. First, on which question types does graph retrieval beat
vector retrieval, and where does it lose (Single-Fact, Multi-Fact, Summary)? Second, what does
each framework cost to index and to query?

The work is done when four things hold:
- **The notebooks produce the results.** Running notebooks `01` to `05` in order writes one
  results table, with one row per system, question type and metric.
- **The table covers quality and cost.** It includes accuracy, summary recall/precision/F1,
  query cost, latency, indexing cost and indexing time.
- **A small subset runs first.** The whole pipeline runs on a subset before anyone pays for the
  full dataset.
- **The key steps are visible.** A learner sees every important step directly in the notebooks.

## Current state

**The repository is empty and has no precedent of its own.**
`/Users/linafaik/Documents/projects/graph-retrieval-bench` contains no files and is not a git
repository. The conventions therefore come from the author's reference repository,
`linafaik08/tabular-foundation-model-evaluation`, which was checked on 2026-09-17:
- **Code layout.** `src/` holds flat modules (`data.py`, `evaluation.py`, `visualization.py`).
  Its `pyproject.toml` declares `[tool.setuptools] packages = ["src"]`, and
  `uv pip install -e .` installs it.
- **Notebook imports.** Notebooks import with `from src.data import ...`.
- **Model calls.** The model calls stay in the notebook, "so that every call is visible where it
  is explained".
- **Charts.** Figures use Plotly.
- **README.** The README has these sections: title, subtitle, author and dates; Objective;
  Project Description (with subsections for the compared models, Data and Code Structure);
  How to Use This Repository? (Requirements, Installation, Setup, Running the Project, Key
  Features Demonstrated); Resources; License; and a closing tagline.
- **Tests.** There is no test suite.

The following external facts were checked on 2026-09-17:

- **Dataset.** The dataset is published on Hugging Face as `Bstwpy/WildGraphBench` (Apache-2.0).
  - *Layout.* It has twelve domain folders with lowercase names, such as `technology` and
    `human_activities`. The questions are in `QA/<domain>/questions.jsonl`. The corpus is in
    `corpus/<domain>/<topic>/reference_pages/*.txt`, with a `references.jsonl` metadata file.
    The Technology corpus is 12.3 MB.
  - *Question fields.* Each question has `question`, `question_type` (a list), `answer`,
    `gold_statements` (for Summary) and `ref_urls`.
  - *Technology counts.* The paper reports 56 Single-Fact, 33 Multi-Fact and 24 Summary
    questions.
- **Official evaluation.** `tools/eval.py` in `BstWPY/WildGraphBench` defines the protocol.
  - *Single-Fact and Multi-Fact.* A binary LLM judge compares each answer with the reference.
  - *Summary.* The judge computes statement recall (coverage), statement precision (accuracy)
    and their F1.
  - *Defaults.* The default judge is `gpt-5-mini`. The script reads each prediction from the
    `pred_answer` field.
- **Paper setup and findings.** The paper builds graphs and answers with `gpt-4o-mini`, and
  judges with `gpt-5-mini`.
  - *Multi-Fact.* Graph methods win; GraphRAG global scored 47.6% against 35.1% for NaiveRAG.
  - *Single-Fact.* NaiveRAG stays competitive.
  - *Summary.* NaiveRAG has the best F1.
- **LightRAG** (`lightrag-hku`).
  - *Query modes.* `QueryParam(mode=...)` accepts `naive | local | global | hybrid | mix`.
  - *Custom functions.* It accepts a custom `llm_model_func` and `embedding_func`.
- **Microsoft GraphRAG** (`graphrag`).
  - *CLI.* The commands are `graphrag init`, `graphrag index` and
    `graphrag query --method local|global`.
  - *Configuration.* `settings.yaml` declares `completion_models` and `embedding_models`, which
    are called through LiteLLM. Each workflow and search mode selects its model through
    `completion_model_id`.

**Still open.** The exact `graphrag.api` Python signatures and the chunking defaults of both
libraries are not shown in the documentation pages that were read. Step 1 downloads the three
papers into `papers/` and checks these points in the source of the pinned library versions.

## Approach

### One way to run the benchmark
The notebooks are the only entry point. `src/` holds small helper functions that are not the
subject of the lesson: downloading data, tracking costs, looping over questions, judging and
plotting. The code worth teaching stays in the notebooks, in line with the reference
repository:
- **NaiveRAG.** Chunking, embedding, cosine top-k and the answer prompt.
- **LightRAG.** The `insert` and `query` calls.
- **GraphRAG.** The `build_index`, `local_search` and `global_search` calls.

### Configuration and run modes
`src/config.py` is the single place a learner edits. It defines:

```python
DOMAIN: str = "technology"
RUN_MODE: RunMode = RunMode.SUBSET          # RunMode.SUBSET | RunMode.FULL
SUBSET_QUESTIONS_PER_TYPE: int = 3          # 9 questions, only the documents they cite
INDEXING_MODEL: str = "gpt-4o-mini"
ANSWER_MODEL: str = "gpt-4o-mini"
EMBEDDING_MODEL: str = "text-embedding-3-small"
JUDGE_MODEL: str = "gpt-5-mini"
CHUNK_SIZE_TOKENS: int = 1200
CHUNK_OVERLAP_TOKENS: int = 100
MAX_CONCURRENT_LLM_CALLS: int = 8
PRICES_USD_PER_MILLION_TOKENS: dict[str, ModelPrice]   # with the date prices were checked
```

**The subset comes first.** `RunMode.SUBSET` keeps a stratified sample of questions and only the
documents those questions cite, so a full end-to-end run costs cents and takes minutes. The
learner then switches to `RunMode.FULL` and reruns the notebooks. All outputs are written under
`outputs/<domain>/<run_mode>/`, so subset and full results never mix. `RunMode` is an enum
rather than a boolean flag, so a third mode could be added later without changing any function
signatures.

### Fairness rules
The four systems share these settings, all read from `config.py`:
- **Models.** Every system uses the same indexing model, answer model and embedding model.
- **Chunking.** Every system uses the same chunk size and overlap.
- **Answer style.** Every system asks for "Multiple Paragraphs".
- **Concurrency.** Every system uses the same concurrency limit.

**Each framework answers with its own native prompt.** As you chose, this means the benchmark
measures each framework as shipped, with a fixed model. It does not isolate retrieval alone.
The README and the notebooks state this limitation.

### Cost and token tracking
Every LLM call and embedding call is recorded by one `UsageLedger` object. The ledger writes one
JSON line per call, with these fields: `system_name`, `phase`, `model`, `prompt_tokens`,
`completion_tokens`, `cost_usd` and `question_id`.

Each system feeds the ledger differently:
- **NaiveRAG.** The notebook calls helper functions that read `response.usage` from the OpenAI
  client.
- **LightRAG.** It receives an `llm_model_func` and an `embedding_func` built by helpers that
  record every call, including the keyword-extraction call LightRAG makes at query time.
- **GraphRAG.** It runs in-process through `graphrag.api`. A LiteLLM success callback records the
  usage of every call it makes.

Cost is computed from token counts and the price table, so the published numbers remain
reproducible when prices change. Indexing time is measured with `time.perf_counter()`.

### Evaluation
The judge prompts and metric formulas are copied into `src/evaluation.py` from WildGraphBench
`tools/eval.py`, pinned to a commit and credited. The judge is `gpt-5-mini`, and every judgment
is cached to disk so that re-running the evaluation notebook costs nothing.

Each accuracy comes with a 95% confidence interval computed with the Wilson score method. The
Wilson score interval is a standard way to put error bars on a proportion. It stays between 0%
and 100% and behaves well on small samples, which matters here because each question type has
only 24 to 56 questions.

### Code conventions
- **Explicit names.** Variables and functions have explicit names, such as
  `retrieve_top_k_chunks` rather than `retrieve` and `question_embedding` rather than `q_emb`.
- **Typed, documented functions.** Every function has typed arguments and a return type. Its
  docstring (Google style) documents each argument, the return value and any side effect.
- **Short comments.** Inline comments are rare and at most two sentences. Explanations belong in
  docstrings.
- **Notebook text.** Notebook markdown uses short sentences in a professional tone and follows a
  story: the question, what we build, what we observe, what it means. Each section has a few
  sentences at most.

### Rejected alternatives
- **A headless script path next to the notebooks.** Two ways to run the benchmark would split the
  teaching material and double the maintenance.
- **Wrapping the framework calls in `src/`.** This would hide the calls the lesson is about. The
  reference repository keeps model calls in the notebook for the same reason.
- **A `src/grbench/` package.** It is unnecessary for a teaching repository, and flat modules
  match the reference repository.
- **LightRAG `mode="naive"` as the baseline.** It would tie the baseline to one framework's prompt
  and storage, and hide what vector RAG actually does.
- **The GraphRAG CLI.** A separate process hides its LLM calls from the cost ledger.
- **Calling the official `eval.py` as a subprocess.** It is exact but opaque to learners. It is
  kept only as a one-off check in Testing.

## Reuse

Nothing in the repository can be reused. The plan relies on the reference repository's
conventions and on these library entry points:

- **Download.** `huggingface_hub.snapshot_download(repo_id="Bstwpy/WildGraphBench",
  repo_type="dataset", allow_patterns=[f"QA/{domain}/*", f"corpus/{domain}/*"])` fetches one
  domain.
- **LightRAG.** We use `LightRAG(working_dir, llm_model_func, embedding_func=EmbeddingFunc(...))`,
  `initialize_storages()`, `ainsert(...)` and `aquery(question, QueryParam(mode="hybrid"))`.
- **GraphRAG setup.** `graphrag init` generates `settings.yaml` and `prompts/`. The edited copy is
  committed under `graphrag_project/`.
- **GraphRAG indexing and search.** We use `graphrag.config.load_config`,
  `graphrag.api.build_index`, `graphrag.api.local_search` and `graphrag.api.global_search`. The
  signatures are to be confirmed in step 1.
- **WildGraphBench evaluation.** The judge prompts and the metric formulas come from
  `tools/eval.py`.
- **Tokenisation.** `tiktoken` is used for chunking and for the corpus statistics.

**New modules and why each is needed:**
- **`src/usage_tracking.py`.** No existing tool counts tokens across the three frameworks.
- **`src/question_runner.py`.** It gives all four systems one resumable loop and one prediction
  format.
- **`src/evaluation.py`.** It holds the judge and the results table.
- **`src/visualization.py`.** It holds the Plotly figures, as in the reference repository.

## Steps

1. **Scaffold the repository and download the papers.**
   - *Files.* `pyproject.toml` (uv, `packages = ["src"]`), `.python-version`, `.gitignore`,
     `.env.example` (`OPENAI_API_KEY=`), `src/__init__.py`, `src/config.py`, and `papers/` with
     three PDFs:
     - WildGraphBench (arXiv 2602.02053).
     - LightRAG (arXiv 2410.05779).
     - GraphRAG, "From Local to Global" (arXiv 2404.16130).
   - *Setup.* Run `git init`, then pin `lightrag-hku` and `graphrag`.
   - *Checks against the pinned versions.* Confirm the `graphrag.api` signatures, whether
     `QueryParam` accepts a per-query model, and both chunking defaults. Record the findings in
     `config.py` docstrings.
   - *Dependencies.* None.
2. **Write the usage tracking helpers.**
   - *File.* `src/usage_tracking.py`.
   - *Class `UsageLedger`.* Takes `ledger_path: Path` and provides `record_call(...)` and
     `summarize_by_system_and_phase() -> pd.DataFrame`.
   - *Cost function.* `compute_call_cost_usd(model: str, prompt_tokens: int,
     completion_tokens: int) -> float`.
   - *OpenAI wrappers.* `create_chat_completion(messages: list[dict], model: str,
     ledger: UsageLedger, system_name: str, phase: Phase, question_id: str | None) -> str` and
     `create_embeddings(texts: list[str], model: str, ledger: UsageLedger, system_name: str,
     phase: Phase) -> np.ndarray`.
   - *Framework adapters.* `build_lightrag_llm_function(...)`,
     `build_lightrag_embedding_function(...)` and `register_litellm_usage_callback(...)`.
   - *Dependencies.* Step 1.
3. **Write the data helpers and notebook 01.**
   - *Files.* `src/data.py` and `notebooks/01_data_preparation.ipynb`.
   - *`download_wildgraphbench_domain(domain: str, raw_data_directory: Path) -> Path`.* Downloads
     one domain.
   - *`load_domain_questions(raw_domain_directory: Path, domain: str) -> pd.DataFrame`.* Adds a
     `question_id`, normalises `question_type` to `single_fact | multi_fact | summary`, and fails
     if the counts per type differ from the expected ones (56/33/24 for Technology).
   - *`load_domain_documents(raw_domain_directory: Path) -> pd.DataFrame`.* Loads the corpus.
   - *`select_question_subset(questions: pd.DataFrame, questions_per_type: int,
     random_seed: int) -> pd.DataFrame`.* Draws the stratified subset.
   - *`select_documents_cited_by_questions(questions: pd.DataFrame,
     documents: pd.DataFrame) -> pd.DataFrame`.* Maps `ref_urls` to documents through
     `references.jsonl`.
   - *`split_text_into_chunks(text: str, chunk_size_tokens: int,
     chunk_overlap_tokens: int) -> list[str]`.* Splits a document into overlapping chunks.
   - *`prepare_run_inputs(config) -> RunInputs`.* Writes `questions.jsonl` and a `corpus/`
     folder of `.txt` files for the active run mode.
   - *Notebook 01.* Shows one question of each type, the corpus size, and why a "wild" corpus is
     hard.
   - *Dependencies.* Step 1.
4. **Build the question runner.**
   - *File.* `src/question_runner.py`.
   - *Function.* `answer_all_questions(answer_function: Callable[[str, str], Awaitable[str]],
     questions: pd.DataFrame, system_name: str, predictions_path: Path,
     ledger: UsageLedger) -> pd.DataFrame`.
   - *Output.* It writes one JSON line per question, with the fields `question_id`,
     `question_type`, `question`, `pred_answer`, `latency_seconds` and `cost_usd`.
   - *Resuming.* It skips questions that already have an answer, so an interrupted run does not
     pay twice.
   - *Dependencies.* Step 2.
5. **Write notebook 02, NaiveRAG.**
   - *File.* `notebooks/02_naive_rag.ipynb`.
   - *Code in the notebook.* The notebook chunks and embeds the corpus, defines
     `retrieve_top_k_chunks` (numpy cosine similarity) and `answer_with_naive_rag`, records the
     indexing time, and runs `answer_all_questions`.
   - *Inspection.* It prints the chunks retrieved for one Multi-Fact question, so the learner
     sees what vector search misses.
   - *Dependencies.* Steps 3 and 4.
6. **Write notebook 03, LightRAG.**
   - *File.* `notebooks/03_lightrag.ipynb`.
   - *Code in the notebook.* The notebook builds the LightRAG instance with the tracked
     functions, inserts the documents, times the indexing, and answers the questions with
     `mode="hybrid"`.
   - *Inspection.* It loads the saved GraphML and shows the node count, the edge count, the
     top entities and one entity description.
   - *Dependencies.* Steps 3 and 4.
7. **Write notebook 04, GraphRAG.**
   - *Files.* `graphrag_project/settings.yaml`, `graphrag_project/prompts/` and
     `notebooks/04_graphrag.ipynb`.
   - *Code in the notebook.* The notebook loads the config, points its input and output at the
     active run folder, registers the LiteLLM callback, and runs `build_index`. It then answers
     every question twice, once with `local_search` and once with `global_search`, using a fixed
     community level.
   - *Inspection.* It shows the entities, relationships and communities, plus one community
     report. It explains that global search is a map-reduce over reports, which makes it the
     most expensive mode per question.
   - *Dependencies.* Steps 3 and 4.
8. **Write the evaluation, visualization and notebook 05.**
   - *Files.* `src/evaluation.py`, `src/visualization.py` and
     `notebooks/05_evaluation_and_results.ipynb`.
   - *Judging.* `judge_fact_answer(...)` and `judge_summary_answer(...)` score single answers,
     and `evaluate_system_predictions(system_name: str, run_directory: Path) -> pd.DataFrame`
     scores a whole system with cached judgments.
   - *Confidence intervals.* `compute_wilson_interval(correct_count: int, total_count: int,
     confidence_level: float) -> tuple[float, float]`.
   - *`build_results_table(run_directory: Path) -> pd.DataFrame`.* Returns a long table with the
     columns `system_name, question_type, metric, value, ci_low, ci_high, question_count`.
     - *Quality metrics.* `accuracy` for Single-Fact and Multi-Fact; `recall`, `precision` and
       `f1` for Summary.
     - *Query metrics, per type.* `query_cost_usd_per_question` and
       `latency_seconds_per_question`.
     - *Indexing metrics.* `indexing_cost_usd`, `indexing_tokens` and `indexing_time_seconds`,
       with `question_type = "all"`.
   - *Outputs.* The table is saved as `results.csv`, alongside `run_manifest.json`, which records
     the models, prices, package versions, dataset revision and run mode.
   - *Notebook 05.* Judges every system, shows the table, plots accuracy by question type and
     cost against accuracy, and closes with the findings.
   - *Dependencies.* Steps 5 to 7.
9. **Write the README, then run the subset and the full benchmark.**
   - *README.* It follows the reference structure:
     - Title, subtitle, author and dates.
     - Objective.
     - Project Description, with the subsections The Four Systems, Data and Code Structure.
     - How to Use This Repository?, with Requirements, Installation, Setup (the API key and the
       expected cost), Running the Project (subset first, then full), and Key Features
       Demonstrated.
     - Resources (papers and repositories), License, and the tagline.
   - *Subset run.* Run notebooks 01–05 with `RunMode.SUBSET` and check the costs and the table.
   - *Full run.* Switch to `RunMode.FULL`, rerun, and commit the full `results.csv` and the
     manifest.
   - *Dependencies.* Step 8.

Resulting layout:

```
papers/                       # WildGraphBench, LightRAG and GraphRAG papers (PDF)
graphrag_project/             # settings.yaml + prompts/ for Microsoft GraphRAG
notebooks/
├── 01_data_preparation.ipynb
├── 02_naive_rag.ipynb
├── 03_lightrag.ipynb
├── 04_graphrag.ipynb
└── 05_evaluation_and_results.ipynb
src/
├── config.py                 # domain, run mode, models, prices, chunking
├── data.py                   # download, normalise, subset, chunk
├── usage_tracking.py         # token and cost ledger + framework adapters
├── question_runner.py        # resumable loop over questions
├── evaluation.py             # WildGraphBench judge, metrics, results table
└── visualization.py          # Plotly figures
outputs/<domain>/<run_mode>/  # inputs, indexes, predictions, judgments, usage (gitignored)
results/<domain>/full/        # results.csv + run_manifest.json (committed)
```

## Risks and edge cases

- **Full-run cost.** The Technology corpus is about 3M tokens, or roughly 2,500 chunks. By rough
  estimate, and not yet measured, graph extraction with `gpt-4o-mini` costs a few dollars to the
  low tens of dollars per graph framework, and takes tens of minutes or more. The subset run
  measures the real cost per chunk before the full run, and notebook 01 prints a projected full
  cost from that measurement.
- **Caches hide real costs.** Both frameworks cache LLM responses, so a rerun on an existing index
  shows almost no cost. Each notebook records whether the index was built fresh, and the results
  table flags indexing costs that come from a reused index.
- **GraphRAG API changes.** GraphRAG's API and configuration change between major versions. The
  version is pinned and `settings.yaml` is committed. If the LiteLLM callback misses calls, the
  fallback is a dedicated OpenAI project key read from the usage dashboard, and the README
  documents it.
- **Unknown question labels.** The raw values of `question_type` have not been seen yet.
  `load_domain_questions` fails loudly on unexpected labels or counts.
- **Unmapped URLs.** Some `ref_urls` may not map to any document. The subset selection skips those
  questions and logs them.
- **Judge variance.** `gpt-5-mini` does not accept a temperature of 0. Cached judgments keep the
  published table stable, and a fresh judging run may move scores by a few points.
- **Small samples.** With 24 Summary questions in the full run, one question moves the score by
  about 4 points. The subset run is for checking the pipeline only; its scores are not
  meaningful, and the notebooks say so.
- **Rate limits.** The shared concurrency limit and the resumable prediction files make `429`
  errors recoverable.

## Testing

The reference repository has no test suite, and this plan follows it. Verification relies on
the pipeline itself:
- **Subset run.** With `RunMode.SUBSET`, all five notebooks run top to bottom. Each of the four
  systems writes one prediction per subset question, the indexing and query costs are above
  zero, and `results.csv` contains every system × question type × metric row.
- **Built-in data checks.** `load_domain_questions` asserts the counts per question type, and
  `answer_all_questions` asserts that every question has a non-empty answer.
- **Protocol check.** On the NaiveRAG predictions, the official WildGraphBench `tools/eval.py`
  must give the same accuracy and F1 as `src/evaluation.py`, allowing for judge variance.
- **Cost check.** After the subset run, the ledger total must match the OpenAI usage dashboard
  for the same time window.

## Out of scope

- **Other retrieval systems.** HippoRAG2, Fast-GraphRAG, BM25 and GraphRAG drift search are not
  included.
- **Tuning.** Each framework uses its shipped settings, apart from the shared fairness settings.
- **Model and domain runs.** Only OpenAI models run on the `technology` domain. The model and the
  domain are both configuration parameters.
- **Headless runs and tooling.** There is no headless script entry point, orchestration layer,
  CI pipeline or unit-test suite.
- **Implementation code.** This document is the plan only.

## Revision log

- Round 1: Applied your feedback:
  - *Configuration.* The domain became a configuration parameter, and a subset-first run mode
    was added.
  - *Structure.* The code now runs through notebooks only, with the key code in the notebooks,
    flat `src/` modules and the reference README structure.
  - *Additions.* Paper downloads, explicit typed signatures and style rules were added, and the
    Wilson interval is now explained.
