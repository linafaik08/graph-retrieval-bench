"""LLM-judge evaluation and results table, following the WildGraphBench protocol.

The prompts and scoring rules are copied from the official evaluator,
https://github.com/BstWPY/WildGraphBench/blob/c334bc806511e029285d208c24c2a8901ed5cf2c/tools/eval.py
(Apache-2.0).

- Single-fact and multi-fact questions: the judge compares the answer with the reference
  answer and returns 1 (correct) or 0 (incorrect).
- Summary questions: the judge splits the answer into atomic statements, then checks
  (a) which gold statements the answer supports (recall, called "coverage" in the paper) and
  (b) which answer statements the gold statements support (precision). F1 combines both.

One deliberate difference: the official script checks precision against a ``answer`` field
that summary questions do not have, which forces precision to zero. Here the gold statements,
joined into a paragraph, serve as the reference text.
"""

import asyncio
import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src import config
from src.question_runner import get_predictions_path, load_predictions
from src.usage_tracking import Phase, UsageLedger, create_chat_completion, load_indexing_report, usage_scope

MAX_STATEMENTS_PER_ANSWER = 20

JUDGE_SYSTEM_PROMPT = """You are a careful grader. You must return ONLY a valid JSON object.
Do not include any explanatory text before or after the JSON.
Do not wrap the JSON in markdown code blocks.
Just return the raw JSON object directly."""

FACT_ANSWER_PROMPT = """Grade whether the CANDIDATE ANSWER is CORRECT (1) or INCORRECT (0) relative to the REFERENCE ANSWER.

Rules:
- Paraphrasing is allowed if meaning is identical.
- Grade 0 if candidate omits/distorts key facts.
- Grade 0 if candidate contradicts reference.
- Grade 1 if candidate matches all key facts.

Return this exact JSON format:
{{"score": 0, "reason": "brief justification"}}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

CANDIDATE ANSWER:
{candidate_answer}"""

STATEMENT_EXTRACTION_PROMPT = """You will be given a QUESTION and a CANDIDATE (model) SUMMARY ANSWER.

Your task is to extract a list of short, atomic factual statements from the SUMMARY ANSWER.

Requirements:
- Each statement must express exactly ONE factual claim (subject + predicate + key objects).
- Statements should be concise declarative sentences (no bullet markers needed).
- Do NOT invent new information that is not present in the SUMMARY ANSWER.
- You may lightly rewrite for clarity, but the meaning must stay the same.
- Return at most {max_statements} statements.
- If the SUMMARY ANSWER is empty or purely non-factual, return an empty list.

QUESTION:
{question}

SUMMARY ANSWER:
{candidate_answer}

Return JSON ONLY in this format:
{{"statements": ["...", "..."]}}"""

STATEMENT_SUPPORT_PROMPT = """You will be given a REFERENCE ANSWER (a short paragraph) and a list of CANDIDATE STATEMENTS.

For EACH candidate statement, decide whether it is fully and factually supported by the REFERENCE ANSWER.

Guidelines:
- Return 1 only if the statement can be clearly and fully inferred from the REFERENCE ANSWER with no contradictions.
- If the statement is partially wrong, missing key constraints, or contradicted, return 0.
- If the REFERENCE ANSWER does not clearly state the fact, return 0.
- Ignore wording differences; focus on factual meaning (entities, numbers, dates, core relations).

REFERENCE ANSWER:
{reference_text}

CANDIDATE STATEMENTS (indexed):
{indexed_statements}

Return JSON ONLY in this format:
{{"scores": [{{"idx": 1, "score": 1}}, {{"idx": 2, "score": 0}}]}}"""


def _parse_json_object(judge_output: str) -> dict[str, Any]:
    """Parse the judge output, tolerating stray text around the JSON object."""
    try:
        return json.loads(judge_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", judge_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


async def _ask_judge(prompt: str, ledger: UsageLedger) -> dict[str, Any]:
    """Send one grading prompt to the judge model and return its JSON answer (empty if invalid)."""
    judge_output = await create_chat_completion(
        messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        model=config.JUDGE_MODEL,
        ledger=ledger,
        reasoning_effort=config.JUDGE_REASONING_EFFORT,
        response_format={"type": "json_object"},
    )
    return _parse_json_object(judge_output)


def _read_binary_score(raw_score: Any) -> int:
    """Convert the judge's score field to 0 or 1, as the official evaluator does."""
    if isinstance(raw_score, str):
        return int(raw_score.strip().lower() in ("1", "true", "correct", "yes"))
    return int(raw_score in (1, True))


async def judge_fact_answer(
    question: str, reference_answer: str, candidate_answer: str, ledger: UsageLedger
) -> dict[str, Any]:
    """Grade a single-fact or multi-fact answer as correct or incorrect.

    Args:
        question: Question text.
        reference_answer: Gold answer from the dataset.
        candidate_answer: Answer produced by a retrieval system.
        ledger: Ledger the judge call is recorded in.

    Returns:
        ``{"correct": 0 or 1, "reason": str}``.
    """
    if not candidate_answer.strip():
        return {"correct": 0, "reason": "Empty answer"}
    judge_answer = await _ask_judge(
        FACT_ANSWER_PROMPT.format(
            question=question, reference_answer=reference_answer, candidate_answer=candidate_answer
        ),
        ledger,
    )
    if not judge_answer:
        return {"correct": 0, "reason": "Judge returned invalid JSON"}
    return {"correct": _read_binary_score(judge_answer.get("score", 0)), "reason": judge_answer.get("reason", "")}


async def extract_answer_statements(question: str, candidate_answer: str, ledger: UsageLedger) -> list[str]:
    """Split an answer into atomic factual statements.

    Args:
        question: Question text, given to the judge as context.
        candidate_answer: Answer produced by a retrieval system.
        ledger: Ledger the judge call is recorded in.

    Returns:
        Up to ``MAX_STATEMENTS_PER_ANSWER`` distinct statements, in the judge's order.
    """
    if not candidate_answer.strip():
        return []
    judge_answer = await _ask_judge(
        STATEMENT_EXTRACTION_PROMPT.format(
            max_statements=MAX_STATEMENTS_PER_ANSWER, question=question, candidate_answer=candidate_answer
        ),
        ledger,
    )
    statements = [str(statement).strip() for statement in judge_answer.get("statements", []) if str(statement).strip()]
    return list(dict.fromkeys(statements))[:MAX_STATEMENTS_PER_ANSWER]


async def check_statements_supported(reference_text: str, statements: list[str], ledger: UsageLedger) -> list[int]:
    """Ask the judge which statements a reference text supports.

    Args:
        reference_text: Text used as evidence.
        statements: Statements to verify.
        ledger: Ledger the judge call is recorded in.

    Returns:
        One flag per statement: 1 if supported, 0 otherwise (also 0 when the judge fails).
    """
    if not statements or not reference_text.strip():
        return [0] * len(statements)
    indexed_statements = "\n".join(f"[{index + 1}] {statement}" for index, statement in enumerate(statements))
    judge_answer = await _ask_judge(
        STATEMENT_SUPPORT_PROMPT.format(reference_text=reference_text, indexed_statements=indexed_statements),
        ledger,
    )
    support_flags = [0] * len(statements)
    for item in judge_answer.get("scores", []):
        try:
            statement_index = int(item["idx"])
            if 1 <= statement_index <= len(statements):
                support_flags[statement_index - 1] = int(float(item.get("score", 0)) >= 0.5)
        except (KeyError, TypeError, ValueError):
            continue
    return support_flags


async def judge_summary_answer(
    question: str, gold_statements: list[str], candidate_answer: str, ledger: UsageLedger
) -> dict[str, Any]:
    """Grade a summary answer at the statement level.

    Args:
        question: Question text.
        gold_statements: Facts the answer should cover, from the dataset.
        candidate_answer: Answer produced by a retrieval system.
        ledger: Ledger the judge calls are recorded in.

    Returns:
        ``recall`` (share of gold statements supported by the answer), ``precision`` (share of
        answer statements supported by the gold statements), ``f1``, ``correct`` (1 only when
        both are 1) and ``answer_statements``.
    """
    answer_statements = await extract_answer_statements(question, candidate_answer, ledger)
    if not gold_statements or not answer_statements:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "correct": 0, "answer_statements": answer_statements}

    gold_support_flags, answer_support_flags = await asyncio.gather(
        check_statements_supported(candidate_answer, gold_statements, ledger),
        check_statements_supported(" ".join(gold_statements), answer_statements, ledger),
    )
    recall = sum(gold_support_flags) / len(gold_statements)
    precision = sum(answer_support_flags) / len(answer_statements)
    f1 = 2 * recall * precision / (recall + precision) if recall > 0 and precision > 0 else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "correct": int(recall >= 1.0 and precision >= 1.0),
        "answer_statements": answer_statements,
    }


def _hash_answer(answer_text: str) -> str:
    """Fingerprint an answer so a changed prediction is judged again."""
    return hashlib.sha1(answer_text.encode()).hexdigest()[:16]


async def evaluate_system_predictions(
    system_name: str,
    questions: pd.DataFrame,
    run_directory: Path,
    ledger: UsageLedger,
    max_concurrent_judgments: int,
) -> pd.DataFrame:
    """Judge every prediction of one system, reusing judgments already on disk.

    Side effect: appends new judgments to ``judgments/<system_name>.jsonl`` in the run folder.

    Args:
        system_name: Retrieval system name.
        questions: Questions of the run, with ``answer`` and ``gold_statements``.
        run_directory: Root folder of the run.
        ledger: Ledger the judge calls are recorded in (phase ``judging``).
        max_concurrent_judgments: Number of questions judged in parallel.

    Returns:
        One row per question with ``question_id``, ``question_type``, ``correct`` and, for
        summary questions, ``recall``, ``precision`` and ``f1``.
    """
    judgments_path = run_directory / "judgments" / f"{system_name}.jsonl"
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    cached_judgments = {}
    if judgments_path.exists():
        for line in judgments_path.read_text().splitlines():
            judgment = json.loads(line)
            cached_judgments[(judgment["question_id"], judgment["answer_hash"])] = judgment

    predictions = load_predictions(get_predictions_path(run_directory, system_name))
    predictions_with_gold = questions.merge(predictions[["question_id", "pred_answer"]], on="question_id", how="inner")
    concurrency_limit = asyncio.Semaphore(max_concurrent_judgments)
    write_lock = asyncio.Lock()

    async def judge_one_prediction(row: pd.Series) -> dict[str, Any]:
        answer_hash = _hash_answer(row["pred_answer"])
        cache_key = (row["question_id"], answer_hash)
        if cache_key in cached_judgments:
            return cached_judgments[cache_key]
        async with concurrency_limit:
            with usage_scope(system_name, Phase.JUDGING, question_id=row["question_id"]):
                if row["question_type"] == "summary":
                    scores = await judge_summary_answer(
                        row["question"], list(row["gold_statements"]), row["pred_answer"], ledger
                    )
                else:
                    scores = await judge_fact_answer(row["question"], row["answer"], row["pred_answer"], ledger)
        judgment = {
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "answer_hash": answer_hash,
            "judge_model": config.JUDGE_MODEL,
            **scores,
        }
        async with write_lock:
            with judgments_path.open("a", encoding="utf-8") as judgments_file:
                judgments_file.write(json.dumps(judgment, ensure_ascii=False) + "\n")
        return judgment

    tasks = [judge_one_prediction(row) for _, row in predictions_with_gold.iterrows()]
    judgments = [
        await task for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"judge {system_name}")
    ]
    return pd.DataFrame(judgments).sort_values("question_id").reset_index(drop=True)


def compute_wilson_interval(
    correct_count: int, total_count: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Confidence interval of an accuracy, using the Wilson score method.

    The Wilson interval stays within [0, 1] and remains reliable on small samples, unlike the
    textbook ``p ± z·sqrt(p(1-p)/n)`` interval.

    Args:
        correct_count: Number of correct answers.
        total_count: Number of graded answers.
        confidence_level: Coverage of the interval.

    Returns:
        ``(lower_bound, upper_bound)``, or ``(nan, nan)`` when ``total_count`` is 0.
    """
    if total_count == 0:
        return float("nan"), float("nan")
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    observed_accuracy = correct_count / total_count
    denominator = 1 + z_score**2 / total_count
    center = (observed_accuracy + z_score**2 / (2 * total_count)) / denominator
    half_width = (
        z_score
        * ((observed_accuracy * (1 - observed_accuracy) + z_score**2 / (4 * total_count)) / total_count) ** 0.5
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _metric_row(
    system_name: str,
    question_type: str,
    metric: str,
    value: float,
    question_count: int,
    confidence_interval: tuple[float, float] = (float("nan"), float("nan")),
) -> dict[str, Any]:
    """Build one row of the long results table."""
    return {
        "system_name": system_name,
        "question_type": question_type,
        "metric": metric,
        "value": value,
        "ci_low": confidence_interval[0],
        "ci_high": confidence_interval[1],
        "question_count": question_count,
    }


def build_results_table(
    judgments_by_system: dict[str, pd.DataFrame], run_directory: Path, ledger: UsageLedger
) -> pd.DataFrame:
    """Assemble quality, cost and time metrics into one long table.

    Args:
        judgments_by_system: Output of ``evaluate_system_predictions`` for each system.
        run_directory: Root folder of the run, used to read predictions and indexing reports.
        ledger: Usage ledger of the run.

    Returns:
        One row per (system_name, question_type, metric) with the columns ``value``,
        ``ci_low``, ``ci_high`` (accuracy only) and ``question_count``. Indexing metrics use
        ``question_type = "all"``.
    """
    usage_records = ledger.load_records()
    query_records = usage_records[usage_records["phase"] == Phase.QUERY.value]
    indexing_records = usage_records[usage_records["phase"] == Phase.INDEXING.value]
    result_rows = []

    for system_name, judgments in judgments_by_system.items():
        predictions = load_predictions(get_predictions_path(run_directory, system_name))
        system_query_records = query_records[query_records["system_name"] == system_name]
        query_usage_by_question = system_query_records.groupby("question_id").agg(
            query_cost_usd=("cost_usd", "sum"),
            query_tokens=("prompt_tokens", "sum"),
            query_completion_tokens=("completion_tokens", "sum"),
        )
        per_question = (
            judgments.merge(predictions[["question_id", "latency_seconds"]], on="question_id")
            .merge(query_usage_by_question, left_on="question_id", right_index=True, how="left")
            .fillna({"query_cost_usd": 0.0, "query_tokens": 0, "query_completion_tokens": 0})
        )
        per_question["query_tokens"] += per_question.pop("query_completion_tokens")

        for question_type, type_rows in [*per_question.groupby("question_type"), ("all", per_question)]:
            question_count = len(type_rows)
            correct_count = int(type_rows["correct"].sum())
            result_rows.append(
                _metric_row(
                    system_name,
                    question_type,
                    "accuracy",
                    correct_count / question_count,
                    question_count,
                    compute_wilson_interval(correct_count, question_count),
                )
            )
            if question_type == "summary":
                for metric in ("recall", "precision", "f1"):
                    result_rows.append(
                        _metric_row(system_name, question_type, metric, type_rows[metric].mean(), question_count)
                    )
            result_rows.append(
                _metric_row(
                    system_name,
                    question_type,
                    "query_cost_usd_per_question",
                    type_rows["query_cost_usd"].mean(),
                    question_count,
                )
            )
            result_rows.append(
                _metric_row(
                    system_name,
                    question_type,
                    "query_tokens_per_question",
                    type_rows["query_tokens"].mean(),
                    question_count,
                )
            )
            result_rows.append(
                _metric_row(
                    system_name,
                    question_type,
                    "latency_seconds_per_question",
                    type_rows["latency_seconds"].mean(),
                    question_count,
                )
            )

        index_name = config.INDEX_NAME_BY_SYSTEM[system_name]
        system_indexing_records = indexing_records[indexing_records["system_name"] == index_name]
        indexing_report = load_indexing_report(run_directory, index_name) or {}
        question_count = len(per_question)
        result_rows.append(
            _metric_row(
                system_name,
                "all",
                "indexing_cost_usd",
                float(system_indexing_records["cost_usd"].sum()),
                question_count,
            )
        )
        result_rows.append(
            _metric_row(
                system_name,
                "all",
                "indexing_tokens",
                float((system_indexing_records["prompt_tokens"] + system_indexing_records["completion_tokens"]).sum()),
                question_count,
            )
        )
        result_rows.append(
            _metric_row(
                system_name,
                "all",
                "indexing_time_seconds",
                indexing_report.get("indexing_time_seconds", float("nan")),
                question_count,
            )
        )
        result_rows.append(
            _metric_row(
                system_name, "all", "query_cost_usd_total", float(per_question["query_cost_usd"].sum()), question_count
            )
        )
    return pd.DataFrame(result_rows)


def pivot_results_table(results_table: pd.DataFrame) -> pd.DataFrame:
    """Reshape the long results table into one row per system and one column per type and metric.

    Args:
        results_table: Output of ``build_results_table``.

    Returns:
        Wide table indexed by ``system_name``, with ``(question_type, metric)`` columns.
    """
    wide_table = results_table.pivot_table(
        index="system_name", columns=["question_type", "metric"], values="value", aggfunc="first"
    )
    ordered_systems = [system for system in config.SYSTEM_NAMES if system in wide_table.index]
    return wide_table.loc[ordered_systems]


def write_run_manifest(results_directory: Path, question_count: int, document_count: int) -> Path:
    """Record everything needed to reproduce the results table.

    Args:
        results_directory: Folder that receives ``run_manifest.json``.
        question_count: Number of evaluated questions.
        document_count: Number of indexed documents.

    Returns:
        Path of the written manifest.
    """
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None
    manifest = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dataset": {"repository_id": config.DATASET_REPOSITORY_ID, "revision": config.DATASET_REVISION},
        "domain": config.DOMAIN,
        "run_mode": config.RUN_MODE.value,
        "question_count": question_count,
        "document_count": document_count,
        "models": {
            "indexing": config.INDEXING_MODEL,
            "answer": config.ANSWER_MODEL,
            "generation_reasoning_effort": config.GENERATION_REASONING_EFFORT,
            "embedding": config.EMBEDDING_MODEL,
            "judge": config.JUDGE_MODEL,
            "judge_reasoning_effort": config.JUDGE_REASONING_EFFORT,
        },
        "prices_checked_on": config.PRICES_CHECKED_ON,
        "prices_usd_per_million_tokens": {
            model: asdict(price) for model, price in config.PRICES_USD_PER_MILLION_TOKENS.items()
        },
        "chunking": {
            "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            "tokenizer_encoding": config.TOKENIZER_ENCODING,
        },
        "retrieval": {
            "top_k_chunks_by_question_type": config.TOP_K_CHUNKS_BY_QUESTION_TYPE,
            "response_type": config.RESPONSE_TYPE,
            "lightrag_query_mode": config.LIGHTRAG_QUERY_MODE,
            "graphrag_community_level": config.GRAPHRAG_COMMUNITY_LEVEL,
        },
        "package_versions": {
            package: version(package) for package in ("lightrag-hku", "graphrag", "openai", "litellm")
        },
    }
    manifest_path = results_directory / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
