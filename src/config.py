"""Benchmark configuration: the single place to edit before running the notebooks.

Every notebook imports its settings from this module, so the four retrieval systems always
share the same models, chunking and retrieval budget. Values that differ from the reference
papers are flagged in the comments next to them.

Reference papers (PDFs in ``papers/``):
    - WildGraphBench (Wang et al., 2026), the benchmark and its evaluation protocol.
    - LightRAG (Guo et al., 2024).
    - From Local to Global, Microsoft GraphRAG (Edge et al., 2024).
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class RunMode(str, Enum):
    """Size of the benchmark run.

    Attributes:
        SUBSET: A few questions per type, indexed on the documents they cite only. Used to
            check the pipeline and measure the cost per document before paying for a full run.
        FULL: Every question of the domain, indexed on the full domain corpus.
    """

    SUBSET = "subset"
    FULL = "full"


@dataclass(frozen=True)
class ModelPrice:
    """Standard-tier OpenAI price of one model, in US dollars per million tokens.

    Attributes:
        input_usd_per_million_tokens: Price of uncached prompt tokens.
        cached_input_usd_per_million_tokens: Price of prompt tokens served from OpenAI's cache.
        output_usd_per_million_tokens: Price of completion tokens, reasoning tokens included.
    """

    input_usd_per_million_tokens: float
    cached_input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


# --------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY: Path = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIRECTORY: Path = PROJECT_ROOT / "outputs"
RESULTS_DIRECTORY: Path = PROJECT_ROOT / "results"
GRAPHRAG_PROJECT_DIRECTORY: Path = PROJECT_ROOT / "graphrag_project"

# --------------------------------------------------------------------------------------------
# Dataset and run size
# --------------------------------------------------------------------------------------------
# Hugging Face dataset id and the revision the results were produced with.
# Pinning the revision keeps the question set identical across reruns.
DATASET_REPOSITORY_ID: str = "Bstwpy/WildGraphBench"
DATASET_REVISION: str = "bdac7ee73d42901ded65cb438a0f7b1aecd5e67b"

# WildGraphBench domain to benchmark, as named on Hugging Face (lowercase).
# The paper averages all 12 domains; this repository studies one domain in depth.
DOMAIN: str = "technology"

# SUBSET first to validate the pipeline cheaply, then FULL for the published results.
# The paper has no equivalent: it runs every question directly.
RUN_MODE: RunMode = RunMode.SUBSET

# Questions drawn per type in SUBSET mode, only among questions whose cited pages exist.
# Three per type keeps the subset indexing cost to a few cents.
SUBSET_QUESTIONS_PER_TYPE: int = 3

# Seed for every random draw (subset sampling), so reruns select the same questions.
RANDOM_SEED: int = 42

# Question counts per type reported in the WildGraphBench paper (Table 1).
# Data preparation fails if the downloaded file does not match them.
EXPECTED_QUESTION_COUNTS_BY_DOMAIN: dict[str, dict[str, int]] = {
    "culture": {"single_fact": 86, "multi_fact": 37, "summary": 32},
    "geography": {"single_fact": 41, "multi_fact": 24, "summary": 33},
    "health": {"single_fact": 76, "multi_fact": 19, "summary": 55},
    "history": {"single_fact": 25, "multi_fact": 1, "summary": 10},
    "human_activities": {"single_fact": 83, "multi_fact": 13, "summary": 44},
    "mathematics": {"single_fact": 21, "multi_fact": 1, "summary": 11},
    "nature": {"single_fact": 18, "multi_fact": 0, "summary": 10},
    "people": {"single_fact": 77, "multi_fact": 32, "summary": 45},
    "philosophy": {"single_fact": 46, "multi_fact": 6, "summary": 18},
    "religion": {"single_fact": 72, "multi_fact": 4, "summary": 30},
    "society": {"single_fact": 66, "multi_fact": 21, "summary": 27},
    "technology": {"single_fact": 56, "multi_fact": 33, "summary": 24},
}

# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
# LLM that builds the indexes: entity extraction, description summaries, community reports.
# Same model as WildGraphBench and LightRAG, and the cheapest per output token of the candidates.
INDEXING_MODEL: str = "gpt-4o-mini"

# LLM that writes the final answer, identical for all four systems.
# WildGraphBench also shares one generator, gpt-4o-mini, across every system it compares.
ANSWER_MODEL: str = "gpt-4o-mini"

# Sampling temperature of the indexing and answer model (0 = most repeatable output).
# The papers do not report it; 0 limits run-to-run variation. Not supported by reasoning models.
GENERATION_TEMPERATURE: float = 0.0

# Embedding model for chunks, entities and relations in every system.
# The papers do not report theirs; text-embedding-3-small is the cheapest OpenAI option.
EMBEDDING_MODEL: str = "text-embedding-3-small"

# Output size of EMBEDDING_MODEL. LightRAG and the vector stores need it up front.
EMBEDDING_DIMENSION: int = 1536

# LLM judge that grades every answer, as in WildGraphBench.
# Kept identical to the paper so the scores stay comparable to its tables.
JUDGE_MODEL: str = "gpt-5-mini"

# Reasoning budget of the judge. gpt-5-mini does not accept temperature 0.
# "low" keeps grading cheap while leaving room to compare facts carefully.
JUDGE_REASONING_EFFORT: str = "low"

# Standard OpenAI prices, checked on 2026-09-17 at https://developers.openai.com/api/docs/pricing.
# Costs are recomputed from token counts, so changing a price here updates every table.
PRICES_CHECKED_ON: str = "2026-09-17"
PRICES_USD_PER_MILLION_TOKENS: dict[str, ModelPrice] = {
    "gpt-5.6-luna": ModelPrice(0.20, 0.02, 1.20),
    "gpt-5-mini": ModelPrice(0.25, 0.025, 2.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.075, 0.60),
    "text-embedding-3-small": ModelPrice(0.02, 0.02, 0.0),
}

# --------------------------------------------------------------------------------------------
# Chunking and retrieval
# --------------------------------------------------------------------------------------------
# Length of one text chunk, in tokens, for all three frameworks.
# WildGraphBench and LightRAG use 1200; the GraphRAG paper used 600 on its own datasets.
CHUNK_SIZE_TOKENS: int = 1200

# Tokens shared by two consecutive chunks, so a fact on a boundary is not cut in half.
# 100 tokens in both WildGraphBench and the GraphRAG paper.
CHUNK_OVERLAP_TOKENS: int = 100

# Tokenizer used to count tokens for chunking (o200k_base is the gpt-4o / gpt-5 encoding).
TOKENIZER_ENCODING: str = "o200k_base"

# Number of chunks retrieved per question type (NaiveRAG top-k, LightRAG chunk_top_k).
# Values copied from WildGraphBench: broad summary questions get a larger budget.
TOP_K_CHUNKS_BY_QUESTION_TYPE: dict[str, int] = {
    "single_fact": 5,
    "multi_fact": 5,
    "summary": 10,
}

# Free-text answer format pasted into every answer prompt, e.g. "Single Paragraph", "Bullet Points".
# "Multiple Paragraphs" is the LightRAG and GraphRAG default; the papers do not report another value.
RESPONSE_TYPE: str = "Multiple Paragraphs"

# Leiden hierarchy level used by GraphRAG local and global search (0 is the coarsest).
# Level 2 is the GraphRAG default and one of the levels (C2) studied in its paper.
GRAPHRAG_COMMUNITY_LEVEL: int = 2

# LightRAG retrieval mode: "hybrid" combines entity-level and relation-level retrieval.
# WildGraphBench evaluates LightRAG in hybrid mode.
LIGHTRAG_QUERY_MODE: str = "hybrid"

# --------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------
# Maximum simultaneous LLM requests per framework, which shapes the indexing wall time.
# The papers do not report it; 8 stays within default OpenAI rate limits.
MAX_CONCURRENT_LLM_CALLS: int = 8

# Questions answered in parallel by one system. Each question may trigger several LLM calls.
# Kept low because GraphRAG global search fans out to many calls per question.
MAX_CONCURRENT_QUESTIONS: int = 4

# Systems compared in the results table, in display order.
SYSTEM_NAMES: tuple[str, ...] = ("naive_rag", "lightrag_hybrid", "graphrag_local", "graphrag_global")

# Index each system reads. Both GraphRAG search modes share one index, so they share its cost.
INDEX_NAME_BY_SYSTEM: dict[str, str] = {
    "naive_rag": "naive_rag",
    "lightrag_hybrid": "lightrag",
    "graphrag_local": "graphrag",
    "graphrag_global": "graphrag",
}


def get_run_directory(domain: str = DOMAIN, run_mode: RunMode = RUN_MODE) -> Path:
    """Return the folder that holds every artefact of one run, creating it if needed.

    Args:
        domain: WildGraphBench domain name.
        run_mode: Size of the run. Subset and full runs are stored side by side.

    Returns:
        Path to ``outputs/<domain>/<run_mode>/``.
    """
    run_directory = OUTPUTS_DIRECTORY / domain / run_mode.value
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def get_results_directory(domain: str = DOMAIN, run_mode: RunMode = RUN_MODE) -> Path:
    """Return the folder that holds the committed results table of one run, creating it if needed.

    Args:
        domain: WildGraphBench domain name.
        run_mode: Size of the run.

    Returns:
        Path to ``results/<domain>/<run_mode>/``.
    """
    results_directory = RESULTS_DIRECTORY / domain / run_mode.value
    results_directory.mkdir(parents=True, exist_ok=True)
    return results_directory
