"""Download, normalise and subset one WildGraphBench domain.

WildGraphBench stores each domain in two folders on Hugging Face:

- ``QA/<domain>/questions.jsonl``: the questions, their type and their ground truth.
- ``corpus/<domain>/<topic>/``: one Wikipedia article (``<topic>.txt``), the web pages it cites
  (``reference_pages/*.txt``) and a ``references.jsonl`` index of those citations.

The retrieval corpus is the set of cited web pages only. The Wikipedia article is excluded,
because the gold answers were written from it.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tiktoken
from huggingface_hub import snapshot_download

from src.config import EXPECTED_QUESTION_COUNTS_BY_DOMAIN, OUTPUTS_DIRECTORY, RunMode, get_run_directory

QUESTION_TYPE_LABELS: dict[str, str] = {
    "single-fact": "single_fact",
    "single_fact": "single_fact",
    "multi-fact": "multi_fact",
    "multi_fact": "multi_fact",
    "summary": "summary",
}
QUESTION_TYPES: tuple[str, ...] = ("single_fact", "multi_fact", "summary")


@dataclass(frozen=True)
class RunInputs:
    """Questions and documents selected for one run, with the files written for the frameworks.

    Attributes:
        questions: One row per question (see ``load_domain_questions``).
        documents: One row per document (see ``load_domain_documents``).
        run_directory: Root folder of every artefact produced by the run.
        corpus_directory: Folder of ``.txt`` files, one per document, read by GraphRAG.
        questions_path: JSONL copy of ``questions``, in the format of the official evaluator.
    """

    questions: pd.DataFrame
    documents: pd.DataFrame
    run_directory: Path
    corpus_directory: Path
    questions_path: Path


def download_wildgraphbench_domain(domain: str, raw_data_directory: Path, repository_id: str, revision: str) -> Path:
    """Download the questions and corpus of one domain from Hugging Face.

    Files already present locally are not downloaded again.

    Args:
        domain: Domain name as used on Hugging Face, for example ``"technology"``.
        raw_data_directory: Local folder that mirrors the dataset repository layout.
        repository_id: Hugging Face dataset id.
        revision: Dataset commit hash, so every run reads identical files.

    Returns:
        ``raw_data_directory``, now containing ``QA/<domain>/`` and ``corpus/<domain>/``.
    """
    snapshot_download(
        repo_id=repository_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"QA/{domain}/*", f"corpus/{domain}/**"],
        local_dir=raw_data_directory,
    )
    return raw_data_directory


def count_tokens(text: str, encoding_name: str) -> int:
    """Count the tokens of a text with a tiktoken encoding.

    Args:
        text: Text to measure.
        encoding_name: tiktoken encoding, for example ``"o200k_base"``.

    Returns:
        Number of tokens.
    """
    return len(tiktoken.get_encoding(encoding_name).encode(text, disallowed_special=()))


def split_text_into_chunks(
    text: str, chunk_size_tokens: int, chunk_overlap_tokens: int, encoding_name: str
) -> list[str]:
    """Split a text into windows of a fixed number of tokens that overlap.

    This is the same sliding-window strategy that LightRAG and GraphRAG apply by default.

    Args:
        text: Document to split.
        chunk_size_tokens: Number of tokens in each chunk.
        chunk_overlap_tokens: Number of tokens repeated at the start of the next chunk.
        encoding_name: tiktoken encoding used to count tokens.

    Returns:
        The chunks, in document order. An empty text gives an empty list.

    Raises:
        ValueError: If the overlap is not smaller than the chunk size.
    """
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(text, disallowed_special=())
    step_tokens = chunk_size_tokens - chunk_overlap_tokens
    chunks = []
    for start_index in range(0, len(token_ids), step_tokens):
        chunks.append(encoding.decode(token_ids[start_index : start_index + chunk_size_tokens]))
        if start_index + chunk_size_tokens >= len(token_ids):
            break
    return chunks


def load_domain_questions(raw_data_directory: Path, domain: str) -> pd.DataFrame:
    """Load the questions of one domain and normalise their labels.

    Args:
        raw_data_directory: Folder passed to ``download_wildgraphbench_domain``.
        domain: Domain name.

    Returns:
        One row per question with the columns ``question_id`` (``<domain>-000`` in file order),
        ``question_type`` (``single_fact``, ``multi_fact`` or ``summary``), ``question``,
        ``answer`` (empty for summary questions), ``gold_statements`` (empty list for fact
        questions) and ``ref_urls``.

    Raises:
        ValueError: If a question has an unknown type, or if the counts per type differ from
            the WildGraphBench paper.
    """
    questions_path = raw_data_directory / "QA" / domain / "questions.jsonl"
    raw_records = [json.loads(line) for line in questions_path.read_text().splitlines() if line.strip()]

    question_rows = []
    for question_index, raw_record in enumerate(raw_records):
        raw_labels = raw_record["question_type"]
        raw_label = raw_labels[0] if isinstance(raw_labels, list) else raw_labels
        if raw_label not in QUESTION_TYPE_LABELS:
            raise ValueError(f"Unknown question type {raw_label!r} in {questions_path}.")
        question_rows.append(
            {
                "question_id": f"{domain}-{question_index:03d}",
                "question_type": QUESTION_TYPE_LABELS[raw_label],
                "question": raw_record["question"],
                "answer": raw_record.get("answer", ""),
                "gold_statements": raw_record.get("gold_statements", []),
                "ref_urls": raw_record.get("ref_urls", []),
            }
        )
    questions = pd.DataFrame(question_rows)

    expected_counts = EXPECTED_QUESTION_COUNTS_BY_DOMAIN.get(domain)
    if expected_counts is not None:
        observed_counts = questions["question_type"].value_counts().to_dict()
        observed_counts = {question_type: observed_counts.get(question_type, 0) for question_type in QUESTION_TYPES}
        if observed_counts != expected_counts:
            raise ValueError(f"Question counts {observed_counts} differ from the paper: {expected_counts}.")
    return questions


def _normalise_title(title: str) -> str:
    """Reduce a title to lowercase letters and digits, the only characters kept in file names."""
    return re.sub(r"[^0-9a-z]+", "", title.lower())


def _build_document_id(topic: str, file_name: str) -> str:
    """Return a short identifier that stays stable across machines and reruns."""
    return "doc-" + hashlib.sha1(f"{topic}/{file_name}".encode()).hexdigest()[:12]


def load_domain_documents(raw_data_directory: Path, domain: str, encoding_name: str) -> pd.DataFrame:
    """Load every cited web page of one domain.

    Args:
        raw_data_directory: Folder passed to ``download_wildgraphbench_domain``.
        domain: Domain name.
        encoding_name: tiktoken encoding used for the ``token_count`` column.

    Returns:
        One row per page with the columns ``document_id``, ``topic``, ``file_name``, ``title``
        (first line of the page), ``text`` and ``token_count``.
    """
    document_rows = []
    for topic_directory in sorted((raw_data_directory / "corpus" / domain).iterdir()):
        for page_path in sorted((topic_directory / "reference_pages").glob("*.txt")):
            text = page_path.read_text(encoding="utf-8", errors="replace")
            first_line = text.split("\n", 1)[0]
            document_rows.append(
                {
                    "document_id": _build_document_id(topic_directory.name, page_path.name),
                    "topic": topic_directory.name,
                    "file_name": page_path.name,
                    "title": first_line.removeprefix("# ").strip() or page_path.stem,
                    "text": text,
                    "token_count": count_tokens(text, encoding_name),
                }
            )
    return pd.DataFrame(document_rows)


def map_reference_urls_to_documents(raw_data_directory: Path, domain: str, documents: pd.DataFrame) -> dict[str, str]:
    """Map every cited URL to the page that stores its content.

    ``references.jsonl`` gives the title of each URL, and page files are named after that
    title with punctuation removed and a length cap. A URL is matched when its normalised title
    equals, or starts with, the normalised file name or the page's first line.

    Args:
        raw_data_directory: Folder passed to ``download_wildgraphbench_domain``.
        domain: Domain name.
        documents: Output of ``load_domain_documents``.

    Returns:
        Dictionary from URL (original and archived) to ``document_id``. URLs whose page was
        never saved by the dataset authors are absent.
    """
    url_to_document_id = {}
    for topic, topic_documents in documents.groupby("topic"):
        document_id_by_key = {}
        for document in topic_documents.itertuples():
            document_id_by_key[_normalise_title(Path(document.file_name).stem)] = document.document_id
            document_id_by_key[_normalise_title(document.title)] = document.document_id

        references_path = raw_data_directory / "corpus" / domain / topic / "references.jsonl"
        for line in references_path.read_text().splitlines():
            reference = json.loads(line)
            normalised_title = _normalise_title(reference["title"])
            document_id = document_id_by_key.get(normalised_title)
            if document_id is None:
                # File names are truncated, so a long title only matches by prefix.
                document_id = next(
                    (
                        value
                        for key, value in document_id_by_key.items()
                        if len(key) >= 40 and normalised_title.startswith(key)
                    ),
                    None,
                )
            if document_id is None:
                continue
            for url in (reference.get("url"), reference.get("archive_url")):
                if url:
                    url_to_document_id[url] = document_id
    return url_to_document_id


def attach_cited_documents(questions: pd.DataFrame, url_to_document_id: dict[str, str]) -> pd.DataFrame:
    """Add the pages cited by each question and flag questions whose evidence is incomplete.

    Args:
        questions: Output of ``load_domain_questions``.
        url_to_document_id: Output of ``map_reference_urls_to_documents``.

    Returns:
        A copy of ``questions`` with two extra columns: ``cited_document_ids`` (list) and
        ``all_citations_available`` (True when every cited URL has a saved page).
    """
    questions_with_citations = questions.copy()
    questions_with_citations["cited_document_ids"] = questions_with_citations["ref_urls"].map(
        lambda urls: sorted({url_to_document_id[url] for url in urls if url in url_to_document_id})
    )
    questions_with_citations["all_citations_available"] = questions_with_citations["ref_urls"].map(
        lambda urls: all(url in url_to_document_id for url in urls)
    )
    return questions_with_citations


def select_question_subset(questions: pd.DataFrame, questions_per_type: int, random_seed: int) -> pd.DataFrame:
    """Draw the same number of questions from each type, among fully supported questions.

    Args:
        questions: Output of ``attach_cited_documents``.
        questions_per_type: Number of questions drawn per type (fewer if a type has fewer).
        random_seed: Seed of the draw.

    Returns:
        The selected questions, sorted by ``question_id``.
    """
    supported_questions = questions[questions["all_citations_available"]]
    sampled_groups = [
        type_questions.sample(n=min(questions_per_type, len(type_questions)), random_state=random_seed)
        for _, type_questions in supported_questions.groupby("question_type")
    ]
    return pd.concat(sampled_groups).sort_values("question_id").reset_index(drop=True)


def select_documents_cited_by_questions(questions: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Keep only the documents cited by at least one of the given questions.

    Args:
        questions: Questions with a ``cited_document_ids`` column.
        documents: Output of ``load_domain_documents``.

    Returns:
        The cited documents.
    """
    cited_document_ids = {document_id for ids in questions["cited_document_ids"] for document_id in ids}
    return documents[documents["document_id"].isin(cited_document_ids)].reset_index(drop=True)


def prepare_run_inputs(
    raw_data_directory: Path,
    domain: str,
    run_mode: RunMode,
    subset_questions_per_type: int,
    random_seed: int,
    encoding_name: str,
) -> RunInputs:
    """Select the questions and documents of a run and write them where the frameworks read them.

    Side effects: writes ``questions.jsonl``, ``documents.parquet`` and one ``.txt`` file per
    document under ``outputs/<domain>/<run_mode>/inputs/``, and the size of the full corpus
    to ``outputs/<domain>/full_corpus_statistics.json``.

    Args:
        raw_data_directory: Folder passed to ``download_wildgraphbench_domain``.
        domain: Domain name.
        run_mode: ``RunMode.SUBSET`` samples questions and keeps their cited pages only;
            ``RunMode.FULL`` keeps everything.
        subset_questions_per_type: Questions per type in subset mode.
        random_seed: Seed of the subset draw.
        encoding_name: tiktoken encoding used to count document tokens.

    Returns:
        The selected questions and documents, and the paths of the written files.
    """
    documents = load_domain_documents(raw_data_directory, domain, encoding_name)
    url_to_document_id = map_reference_urls_to_documents(raw_data_directory, domain, documents)
    questions = attach_cited_documents(load_domain_questions(raw_data_directory, domain), url_to_document_id)
    full_corpus_statistics = {
        "question_count": len(questions),
        "document_count": len(documents),
        "token_count": int(documents["token_count"].sum()),
    }
    full_corpus_statistics_path = OUTPUTS_DIRECTORY / domain / "full_corpus_statistics.json"
    full_corpus_statistics_path.parent.mkdir(parents=True, exist_ok=True)
    full_corpus_statistics_path.write_text(json.dumps(full_corpus_statistics, indent=2))

    if run_mode is RunMode.SUBSET:
        questions = select_question_subset(questions, subset_questions_per_type, random_seed)
        documents = select_documents_cited_by_questions(questions, documents)

    run_directory = get_run_directory(domain, run_mode)
    inputs_directory = run_directory / "inputs"
    corpus_directory = inputs_directory / "corpus"
    corpus_directory.mkdir(parents=True, exist_ok=True)
    for stale_file in corpus_directory.glob("*.txt"):
        stale_file.unlink()
    for document in documents.itertuples():
        (corpus_directory / f"{document.document_id}.txt").write_text(document.text, encoding="utf-8")

    questions_path = inputs_directory / "questions.jsonl"
    questions.to_json(questions_path, orient="records", lines=True, force_ascii=False)
    documents.to_parquet(inputs_directory / "documents.parquet", index=False)

    return RunInputs(
        questions=questions,
        documents=documents,
        run_directory=run_directory,
        corpus_directory=corpus_directory,
        questions_path=questions_path,
    )


def load_full_corpus_statistics(domain: str) -> dict[str, int]:
    """Read the size of the full domain, as written by ``prepare_run_inputs``.

    Args:
        domain: Domain name.

    Returns:
        ``question_count``, ``document_count`` and ``token_count`` of the full domain.
    """
    return json.loads((OUTPUTS_DIRECTORY / domain / "full_corpus_statistics.json").read_text())


def load_run_inputs(domain: str, run_mode: RunMode) -> RunInputs:
    """Reload the inputs written by ``prepare_run_inputs``, without touching the raw data.

    Args:
        domain: Domain name.
        run_mode: Size of the run.

    Returns:
        The questions and documents of the run.

    Raises:
        FileNotFoundError: If notebook 01 has not been run for this domain and run mode.
    """
    run_directory = get_run_directory(domain, run_mode)
    inputs_directory = run_directory / "inputs"
    questions_path = inputs_directory / "questions.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"{questions_path} is missing. Run notebooks/01_data_preparation.ipynb first.")
    return RunInputs(
        questions=pd.read_json(questions_path, orient="records", lines=True, dtype={"answer": str}),
        documents=pd.read_parquet(inputs_directory / "documents.parquet"),
        run_directory=run_directory,
        corpus_directory=inputs_directory / "corpus",
        questions_path=questions_path,
    )
