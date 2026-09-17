"""Token and cost accounting shared by the three frameworks.

Every LLM and embedding call of the benchmark ends up in one ``UsageLedger``, a JSONL file
with one line per call. Calls reach the ledger through three routes:

- NaiveRAG and the judge call ``create_chat_completion`` and ``create_embeddings`` directly.
- LightRAG receives the functions built by ``build_lightrag_llm_function`` and
  ``build_lightrag_embedding_function``, which wrap the same two helpers.
- GraphRAG sends its calls through LiteLLM, whose callbacks are routed to the ledger by
  ``register_litellm_usage_callback``.

The ledger needs to know which system, phase and question a call belongs to. The
``usage_scope`` context manager declares it once around a block of code, so the frameworks do
not need to pass that information along.
"""

import contextvars
import json
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import litellm
import numpy as np
import pandas as pd
from lightrag.utils import EmbeddingFunc
from litellm.integrations.custom_logger import CustomLogger
from openai import AsyncOpenAI

from src.config import PRICES_USD_PER_MILLION_TOKENS


class Phase(str, Enum):
    """Stage of the benchmark an LLM call belongs to."""

    INDEXING = "indexing"
    QUERY = "query"
    JUDGING = "judging"


@dataclass(frozen=True)
class UsageScope:
    """Attribution attached to every call recorded while the scope is active.

    Attributes:
        system_name: Retrieval system, for example ``"lightrag_hybrid"``.
        phase: Benchmark stage.
        question_id: Question being answered or judged, if any.
    """

    system_name: str
    phase: Phase
    question_id: str | None = None


@dataclass(frozen=True)
class LlmCallRecord:
    """One line of the usage ledger.

    Attributes:
        system_name: Retrieval system that made the call.
        phase: Benchmark stage, as a string.
        question_id: Question the call served, or None for indexing.
        model: Model name as billed by OpenAI.
        prompt_tokens: Input tokens, cached ones included.
        cached_prompt_tokens: Input tokens served from OpenAI's prompt cache.
        completion_tokens: Output tokens, reasoning tokens included.
        cost_usd: Cost computed from ``src.config.PRICES_USD_PER_MILLION_TOKENS``.
        recorded_at: UTC timestamp in ISO format.
    """

    system_name: str
    phase: str
    question_id: str | None
    model: str
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    recorded_at: str


_active_scope_in_context: contextvars.ContextVar[UsageScope | None] = contextvars.ContextVar(
    "active_usage_scope", default=None
)
# Fallback for framework worker threads that do not inherit the caller's context.
_active_scope_in_process: UsageScope | None = None


@contextmanager
def usage_scope(system_name: str, phase: Phase, question_id: str | None = None) -> Iterator[UsageScope]:
    """Attribute every LLM call made inside the ``with`` block to a system, phase and question.

    Scopes can be nested: the question runner opens one per question inside the scope of the
    whole query phase.

    Args:
        system_name: Retrieval system name.
        phase: Benchmark stage.
        question_id: Question being processed, if any.

    Yields:
        The active scope.
    """
    global _active_scope_in_process
    scope = UsageScope(system_name=system_name, phase=phase, question_id=question_id)
    context_token = _active_scope_in_context.set(scope)
    previous_process_scope = _active_scope_in_process
    if question_id is None:
        _active_scope_in_process = scope
    try:
        yield scope
    finally:
        _active_scope_in_context.reset(context_token)
        _active_scope_in_process = previous_process_scope


def get_active_usage_scope() -> UsageScope:
    """Return the scope of the current call.

    Returns:
        The innermost scope of the current async task, or the last process-wide scope when
        the call runs in a worker thread.

    Raises:
        RuntimeError: If a call is made outside any ``usage_scope`` block.
    """
    scope = _active_scope_in_context.get() or _active_scope_in_process
    if scope is None:
        raise RuntimeError("LLM call made outside a usage_scope block; its cost cannot be attributed.")
    return scope


def compute_call_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0
) -> float:
    """Compute the cost of one call from its token counts.

    Args:
        model: Model name. A provider prefix such as ``"openai/"`` is ignored.
        prompt_tokens: Input tokens, cached ones included.
        completion_tokens: Output tokens.
        cached_prompt_tokens: Part of ``prompt_tokens`` served from OpenAI's cache.

    Returns:
        Cost in US dollars.

    Raises:
        KeyError: If the model has no entry in the price table.
    """
    price = PRICES_USD_PER_MILLION_TOKENS[model.split("/")[-1]]
    uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens
    return (
        uncached_prompt_tokens * price.input_usd_per_million_tokens
        + cached_prompt_tokens * price.cached_input_usd_per_million_tokens
        + completion_tokens * price.output_usd_per_million_tokens
    ) / 1_000_000


class UsageLedger:
    """Append-only JSONL log of every LLM call, safe to use from several threads.

    Args:
        ledger_path: File the records are appended to. Parent folders are created.
    """

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def record_call(
        self, model: str, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0
    ) -> LlmCallRecord:
        """Append one call, attributed to the active ``usage_scope``.

        Args:
            model: Model name.
            prompt_tokens: Input tokens, cached ones included.
            completion_tokens: Output tokens.
            cached_prompt_tokens: Part of ``prompt_tokens`` served from OpenAI's cache.

        Returns:
            The written record.
        """
        scope = get_active_usage_scope()
        record = LlmCallRecord(
            system_name=scope.system_name,
            phase=scope.phase.value,
            question_id=scope.question_id,
            model=model.split("/")[-1],
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=compute_call_cost_usd(model, prompt_tokens, completion_tokens, cached_prompt_tokens),
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._write_lock, self.ledger_path.open("a", encoding="utf-8") as ledger_file:
            ledger_file.write(json.dumps(asdict(record)) + "\n")
        return record

    def discard_records(self, system_name: str, phase: Phase) -> int:
        """Delete the records of one system and phase, before an index is rebuilt from scratch.

        Args:
            system_name: Retrieval system or index name.
            phase: Benchmark stage.

        Returns:
            Number of records removed.
        """
        records = self.load_records()
        discarded = (records["system_name"] == system_name) & (records["phase"] == phase.value)
        with self._write_lock:
            records[~discarded].to_json(self.ledger_path, orient="records", lines=True, force_ascii=False)
        return int(discarded.sum())

    def load_records(self) -> pd.DataFrame:
        """Read every recorded call.

        Returns:
            One row per call, with the fields of ``LlmCallRecord``. Empty if nothing was recorded.
        """
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            return pd.DataFrame(columns=list(LlmCallRecord.__dataclass_fields__))
        return pd.read_json(self.ledger_path, lines=True, dtype={"question_id": str})

    def summarize_by_system_and_phase(self) -> pd.DataFrame:
        """Total calls, tokens and cost per system, phase and model.

        Returns:
            One row per (system_name, phase, model) with ``call_count``, token sums and ``cost_usd``.
        """
        records = self.load_records()
        return (
            records.groupby(["system_name", "phase", "model"], as_index=False)
            .agg(
                call_count=("cost_usd", "size"),
                prompt_tokens=("prompt_tokens", "sum"),
                completion_tokens=("completion_tokens", "sum"),
                cost_usd=("cost_usd", "sum"),
            )
            .sort_values(["system_name", "phase", "model"])
        )

    def total_cost_usd(self, system_name: str, phase: Phase) -> float:
        """Total cost of one system in one phase.

        Args:
            system_name: Retrieval system name.
            phase: Benchmark stage.

        Returns:
            Cost in US dollars (0 if nothing was recorded).
        """
        records = self.load_records()
        selected = records[(records["system_name"] == system_name) & (records["phase"] == phase.value)]
        return float(selected["cost_usd"].sum())


_openai_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """Return a shared async OpenAI client that reads ``OPENAI_API_KEY`` from the environment."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(max_retries=6)
    return _openai_client


async def create_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    ledger: UsageLedger,
    reasoning_effort: str,
    response_format: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
) -> str:
    """Call the OpenAI Chat Completions API and record the usage.

    Args:
        messages: Chat messages, each with a ``role`` and a ``content``.
        model: OpenAI model name.
        ledger: Ledger the call is recorded in, under the active ``usage_scope``.
        reasoning_effort: Reasoning budget of the model, for example ``"none"`` or ``"low"``.
        response_format: Optional structured-output setting, for example ``{"type": "json_object"}``.
        max_completion_tokens: Optional cap on output tokens, reasoning included.

    Returns:
        The text of the first choice (empty string if the model returned none).
    """
    optional_arguments: dict[str, Any] = {}
    if response_format is not None:
        optional_arguments["response_format"] = response_format
    if max_completion_tokens is not None:
        optional_arguments["max_completion_tokens"] = max_completion_tokens
    response = await get_openai_client().chat.completions.create(
        model=model, messages=messages, reasoning_effort=reasoning_effort, **optional_arguments
    )
    cached_prompt_tokens = 0
    if response.usage.prompt_tokens_details is not None:
        cached_prompt_tokens = response.usage.prompt_tokens_details.cached_tokens or 0
    ledger.record_call(
        model=model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
    )
    return response.choices[0].message.content or ""


async def create_embeddings(texts: list[str], model: str, ledger: UsageLedger, batch_size: int = 256) -> np.ndarray:
    """Embed texts with the OpenAI Embeddings API and record the usage.

    Args:
        texts: Texts to embed.
        model: OpenAI embedding model name.
        ledger: Ledger the calls are recorded in, under the active ``usage_scope``.
        batch_size: Number of texts sent per request.

    Returns:
        Array of shape ``(len(texts), embedding_dimension)``, in the order of ``texts``.
    """
    embedding_batches = []
    for batch_start in range(0, len(texts), batch_size):
        response = await get_openai_client().embeddings.create(
            model=model, input=texts[batch_start : batch_start + batch_size]
        )
        ledger.record_call(model=model, prompt_tokens=response.usage.prompt_tokens, completion_tokens=0)
        embedding_batches.append(np.array([item.embedding for item in response.data], dtype=np.float32))
    return np.vstack(embedding_batches)


def build_lightrag_llm_function(
    ledger: UsageLedger, model: str, reasoning_effort: str
) -> Callable[..., Awaitable[str]]:
    """Build the ``llm_model_func`` LightRAG calls for extraction, keywords and answers.

    LightRAG calls it as ``func(prompt, system_prompt=..., history_messages=..., **kwargs)``.
    Of the extra keyword arguments, only ``response_format`` and ``max_tokens`` matter here;
    the others (``hashing_kv``, ``enable_cot``, ``stream``) are ignored.

    Args:
        ledger: Ledger every call is recorded in.
        model: OpenAI model name.
        reasoning_effort: Reasoning budget of the model.

    Returns:
        An async function with the signature LightRAG expects.
    """

    async def lightrag_llm_function(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **lightrag_keyword_arguments: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        return await create_chat_completion(
            messages=messages,
            model=model,
            ledger=ledger,
            reasoning_effort=reasoning_effort,
            response_format=lightrag_keyword_arguments.get("response_format"),
            max_completion_tokens=lightrag_keyword_arguments.get("max_tokens"),
        )

    return lightrag_llm_function


def build_lightrag_embedding_function(ledger: UsageLedger, model: str, embedding_dimension: int) -> EmbeddingFunc:
    """Build the ``embedding_func`` LightRAG uses for chunks, entities and relations.

    Args:
        ledger: Ledger every call is recorded in.
        model: OpenAI embedding model name.
        embedding_dimension: Size of the vectors returned by ``model``.

    Returns:
        A LightRAG ``EmbeddingFunc`` wrapping ``create_embeddings``.
    """

    async def lightrag_embedding_function(texts: list[str]) -> np.ndarray:
        return await create_embeddings(texts, model=model, ledger=ledger)

    return EmbeddingFunc(
        embedding_dim=embedding_dimension, max_token_size=8192, model_name=model, func=lightrag_embedding_function
    )


class _LiteLlmLedgerLogger(CustomLogger):
    """LiteLLM callback that copies the usage of every successful call into a ledger."""

    def __init__(self, ledger: UsageLedger) -> None:
        super().__init__()
        self.ledger = ledger

    def _record_response(self, call_arguments: dict[str, Any], response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_token_details = getattr(usage, "prompt_tokens_details", None)
        self.ledger.record_call(
            model=call_arguments.get("model", "unknown"),
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_prompt_tokens=getattr(prompt_token_details, "cached_tokens", 0) or 0,
        )

    def log_success_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        self._record_response(kwargs, response_obj)

    async def async_log_success_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._record_response(kwargs, response_obj)


def register_litellm_usage_callback(ledger: UsageLedger) -> None:
    """Record every LiteLLM call (hence every GraphRAG call) in the ledger.

    LiteLLM fires exactly one success event per call: the synchronous hook for ``completion``
    and the asynchronous hook for ``acompletion``, streamed or not. Calls answered from
    GraphRAG's own cache never reach LiteLLM and are therefore not billed.

    Side effect: replaces ``litellm.callbacks`` with a single ledger logger, so calling this
    function twice does not double-count.

    Args:
        ledger: Ledger the calls are recorded in, under the active ``usage_scope``.
    """
    litellm.callbacks = [_LiteLlmLedgerLogger(ledger)]


def get_ledger_path(run_directory: Path) -> Path:
    """Return the ledger file shared by every notebook of a run.

    Args:
        run_directory: Root folder of the run.

    Returns:
        Path to ``usage/ledger.jsonl`` inside the run folder.
    """
    return run_directory / "usage" / "ledger.jsonl"


def save_indexing_report(
    run_directory: Path, index_name: str, indexing_time_seconds: float, document_count: int, corpus_token_count: int
) -> Path:
    """Store the wall time and size of an index build.

    Args:
        run_directory: Root folder of the run.
        index_name: Index name (``naive_rag``, ``lightrag`` or ``graphrag``).
        indexing_time_seconds: Wall time of the build.
        document_count: Number of indexed documents.
        corpus_token_count: Number of indexed tokens.

    Returns:
        Path of the written JSON file, ``indexing/<index_name>.json``.
    """
    report_path = run_directory / "indexing" / f"{index_name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "index_name": index_name,
        "indexing_time_seconds": round(indexing_time_seconds, 1),
        "document_count": document_count,
        "corpus_token_count": corpus_token_count,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


def load_indexing_report(run_directory: Path, index_name: str) -> dict[str, Any] | None:
    """Read the report written by ``save_indexing_report``.

    Args:
        run_directory: Root folder of the run.
        index_name: Index name.

    Returns:
        The report, or None if the index was never built in this run.
    """
    report_path = run_directory / "indexing" / f"{index_name}.json"
    return json.loads(report_path.read_text()) if report_path.exists() else None


def project_full_run_cost_usd(subset_cost_usd: float, subset_token_count: int, full_token_count: int) -> float:
    """Extrapolate an indexing cost measured on the subset to the full corpus.

    Indexing cost grows roughly linearly with the number of corpus tokens, since every chunk
    goes through the same extraction prompts.

    Args:
        subset_cost_usd: Indexing cost measured on the subset.
        subset_token_count: Corpus tokens in the subset.
        full_token_count: Corpus tokens in the full domain.

    Returns:
        Estimated full indexing cost in US dollars.
    """
    return subset_cost_usd * full_token_count / subset_token_count
