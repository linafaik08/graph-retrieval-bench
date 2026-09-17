"""Configuration and table loading for Microsoft GraphRAG.

These helpers hide two pieces of plumbing that are not the subject of the lesson:

- ``settings.yaml`` uses relative paths, and GraphRAG resolves them against the working
  directory. The loader pins every folder and prompt to an absolute path of the current run.
- The GraphRAG CLI loads its index tables with ``asyncio.run``, which fails inside Jupyter.
  ``load_graphrag_index_tables`` reads them with ``await`` instead.
"""

import os
from pathlib import Path

import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

from src.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    GENERATION_TEMPERATURE,
    GRAPHRAG_PROJECT_DIRECTORY,
    INDEXING_MODEL,
    MAX_CONCURRENT_LLM_CALLS,
    TOKENIZER_ENCODING,
)

GRAPHRAG_TABLE_NAMES: tuple[str, ...] = (
    "entities",
    "relationships",
    "communities",
    "community_reports",
    "text_units",
)


def load_graphrag_config(corpus_directory: Path, index_directory: Path) -> GraphRagConfig:
    """Load ``graphrag_project/settings.yaml`` and point it at the current run.

    The models, chunking and concurrency are taken from ``src/config.py`` so that GraphRAG
    shares the settings of the other systems.

    Args:
        corpus_directory: Folder of ``.txt`` documents to index.
        index_directory: Folder that receives the index, its cache, logs and vector store.

    Returns:
        The validated GraphRAG configuration.

    Raises:
        graphrag_common.config.ConfigParsingError: If ``OPENAI_API_KEY`` is not set.
    """
    prompts_directory = GRAPHRAG_PROJECT_DIRECTORY / "prompts"
    configuration_overrides = {
        "completion_models": {
            "default_completion_model": {
                "model": INDEXING_MODEL,
                "call_args": {"temperature": GENERATION_TEMPERATURE},
            }
        },
        "embedding_models": {"default_embedding_model": {"model": EMBEDDING_MODEL}},
        "concurrent_requests": MAX_CONCURRENT_LLM_CALLS,
        "chunking": {
            "size": CHUNK_SIZE_TOKENS,
            "overlap": CHUNK_OVERLAP_TOKENS,
            "encoding_model": TOKENIZER_ENCODING,
        },
        "input_storage": {"base_dir": str(corpus_directory)},
        "output_storage": {"base_dir": str(index_directory / "output")},
        "cache": {"storage": {"base_dir": str(index_directory / "cache")}},
        "reporting": {"base_dir": str(index_directory / "logs")},
        "vector_store": {"db_uri": str(index_directory / "output" / "lancedb"), "vector_size": EMBEDDING_DIMENSION},
        "extract_graph": {"prompt": str(prompts_directory / "extract_graph.txt")},
        "summarize_descriptions": {"prompt": str(prompts_directory / "summarize_descriptions.txt")},
        "extract_claims": {"prompt": str(prompts_directory / "extract_claims.txt")},
        "community_reports": {
            "graph_prompt": str(prompts_directory / "community_report_graph.txt"),
            "text_prompt": str(prompts_directory / "community_report_text.txt"),
        },
        "local_search": {"prompt": str(prompts_directory / "local_search_system_prompt.txt")},
        "global_search": {
            "map_prompt": str(prompts_directory / "global_search_map_system_prompt.txt"),
            "reduce_prompt": str(prompts_directory / "global_search_reduce_system_prompt.txt"),
            "knowledge_prompt": str(prompts_directory / "global_search_knowledge_system_prompt.txt"),
        },
    }
    # load_config changes the working directory to the settings folder; it is restored here.
    original_working_directory = Path.cwd()
    try:
        return load_config(GRAPHRAG_PROJECT_DIRECTORY, cli_overrides=configuration_overrides)
    finally:
        os.chdir(original_working_directory)


async def load_graphrag_index_tables(config: GraphRagConfig) -> dict[str, pd.DataFrame]:
    """Read the tables written by ``graphrag.api.build_index``.

    Args:
        config: Configuration returned by ``load_graphrag_config``.

    Returns:
        Dictionary with the ``entities``, ``relationships``, ``communities``,
        ``community_reports`` and ``text_units`` tables.
    """
    table_provider = create_table_provider(config.table_provider, storage=create_storage(config.output_storage))
    data_reader = DataReader(table_provider)
    return {table_name: await getattr(data_reader, table_name)() for table_name in GRAPHRAG_TABLE_NAMES}
