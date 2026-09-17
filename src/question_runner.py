"""Answer every question of a run with one retrieval system, with resume support.

Each notebook defines an answer function for its own system and hands it to
``answer_all_questions``. Predictions are appended to
``outputs/<domain>/<run_mode>/predictions/<system_name>.jsonl`` as soon as they are produced, so
an interrupted run restarts where it stopped instead of paying again for answered questions.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.usage_tracking import Phase, usage_scope

AnswerFunction = Callable[[str, str], Awaitable[str]]
"""Async function ``(question, question_type) -> answer`` implemented by each notebook."""


def get_predictions_path(run_directory: Path, system_name: str) -> Path:
    """Return the prediction file of one system.

    Args:
        run_directory: Root folder of the run.
        system_name: Retrieval system name.

    Returns:
        Path to ``predictions/<system_name>.jsonl`` inside the run folder.
    """
    return run_directory / "predictions" / f"{system_name}.jsonl"


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Read a prediction file, keeping the latest attempt of each question.

    Args:
        predictions_path: File written by ``answer_all_questions``.

    Returns:
        One row per question with ``question_id``, ``question_type``, ``question``,
        ``pred_answer``, ``latency_seconds``, ``error`` and ``answered_at``. Empty if the file
        does not exist.
    """
    if not predictions_path.exists():
        return pd.DataFrame(
            columns=[
                "question_id",
                "question_type",
                "question",
                "pred_answer",
                "latency_seconds",
                "error",
                "answered_at",
            ]
        )
    predictions = pd.read_json(
        predictions_path, lines=True, dtype={"question_id": str, "question": str, "pred_answer": str}
    )
    return predictions.drop_duplicates("question_id", keep="last").reset_index(drop=True)


async def answer_all_questions(
    answer_function: AnswerFunction,
    questions: pd.DataFrame,
    system_name: str,
    run_directory: Path,
    max_concurrent_questions: int,
) -> pd.DataFrame:
    """Answer every question that has no valid prediction yet.

    Each question runs inside its own ``usage_scope``, so the ledger attributes every LLM call
    to the question that triggered it. A failed question is stored with its error message and
    retried on the next call.

    Args:
        answer_function: System-specific function returning the answer to one question.
        questions: Questions of the run (``question_id``, ``question_type``, ``question``).
        system_name: Retrieval system name, used for the ledger and the prediction file.
        run_directory: Root folder of the run.
        max_concurrent_questions: Number of questions answered in parallel.

    Returns:
        The predictions of every question of ``questions``.

    Raises:
        RuntimeError: If some questions still have no answer after this pass. Running the
            function again retries them.
    """
    predictions_path = get_predictions_path(run_directory, system_name)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    previous_predictions = load_predictions(predictions_path)
    answered_question_ids = set(
        previous_predictions.loc[
            previous_predictions["pred_answer"].fillna("").str.strip().ne("") & previous_predictions["error"].isna(),
            "question_id",
        ]
    )
    pending_questions = questions[~questions["question_id"].isin(answered_question_ids)]
    already_answered_count = len(questions) - len(pending_questions)
    print(f"{system_name}: {already_answered_count} answered, {len(pending_questions)} to go.")

    concurrency_limit = asyncio.Semaphore(max_concurrent_questions)
    write_lock = asyncio.Lock()

    async def answer_one_question(question_row: pd.Series) -> None:
        async with concurrency_limit:
            start_time = time.perf_counter()
            answer_text, error_message = "", None
            with usage_scope(system_name, Phase.QUERY, question_id=question_row["question_id"]):
                try:
                    answer_text = await answer_function(question_row["question"], question_row["question_type"])
                except Exception as error:  # noqa: BLE001 - stored and retried on the next run
                    error_message = f"{type(error).__name__}: {error}"
            if not error_message and not answer_text.strip():
                error_message = "Empty answer"
            prediction = {
                "question_id": question_row["question_id"],
                "question_type": question_row["question_type"],
                "question": question_row["question"],
                "pred_answer": answer_text,
                "latency_seconds": round(time.perf_counter() - start_time, 3),
                "error": error_message,
                "answered_at": datetime.now(timezone.utc).isoformat(),
            }
            async with write_lock:
                with predictions_path.open("a", encoding="utf-8") as predictions_file:
                    predictions_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    tasks = [answer_one_question(question_row) for _, question_row in pending_questions.iterrows()]
    for finished_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=system_name):
        await finished_task

    predictions = load_predictions(predictions_path)
    predictions = predictions[predictions["question_id"].isin(questions["question_id"])]
    failed_predictions = predictions[predictions["error"].notna()]
    if len(failed_predictions) or len(predictions) < len(questions):
        raise RuntimeError(
            f"{len(questions) - len(predictions) + len(failed_predictions)} questions have no answer. "
            f"First error: {failed_predictions['error'].iloc[0] if len(failed_predictions) else 'missing'}. "
            "Run the cell again to retry them."
        )
    return predictions.reset_index(drop=True)
