import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import sys
import hashlib
from collections import Counter
import math
import logging

# Ensure sibling 'utils' package is importable when running as script via path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds src/app

from utils.params_helper import load_params

import psycopg2
from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import execute_values

import spacy
from spacy.language import Language

import mlflow
from utils.mlflow_helper import mlflow_connect
from utils.mlflow_helper import get_mlflow_experiment_name
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Maximum text length for spaCy processing (stay under 1M limit)
# Parser and NER require ~1GB RAM per 100K chars, so 900K = ~9GB RAM per chunk
MAX_SPACY_LENGTH = 900000

# Maximum sentence length in bytes for PostgreSQL btree index (limit is 2704 bytes)
# Using 2500 as safe margin to account for UTF-8 encoding overhead
MAX_SENTENCE_BYTES = 2500


def compute_hash_id(text: str, prefix: str) -> str:
    """Compute MD5-based hash ID with namespace prefix (matches LinearRAG convention)."""
    return prefix + hashlib.md5(text.encode("utf-8")).hexdigest()


def _connect(database_url: str) -> PGConnection:
    """Connect to Postgres database."""
    return psycopg2.connect(database_url)


class LinearRAGIndexer:
    """
    Build knowledge graph for LinearRAG by extracting entities and sentences,
    then creating weighted graph edges between passages, entities, and sentences.
    """

    def __init__(self, conn: PGConnection, nlp: Language, batch_size: int = 32):
        self.conn = conn
        self.nlp = nlp
        self.batch_size = batch_size

        # Statistics for MLflow logging
        self.stats = {
            "passages_processed": 0,
            "passages_skipped": 0,
            "entities_extracted": 0,
            "sentences_extracted": 0,
            "sentences_filtered_oversized": 0,
            "unique_entities": 0,
            "unique_sentences": 0,
            "graph_nodes_created": 0,
            "graph_edges_created": 0,
        }

    def process_chunks(self) -> None:
        """
        Main processing pipeline:
        1. Load passages from document_chunk table
        2. Extract entities and sentences using spaCy NER
        3. Insert into lr_* tables with hash IDs
        4. Build graph structure with weighted edges
        """
        # Load passages from database
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, page_ref, content, chunk_hash_id
                FROM document_chunk
                WHERE chunk_hash_id IS NOT NULL
                ORDER BY title, page_ref
                """
            )
            db_passages = cur.fetchall()

        if not db_passages:
            logger.warning(
                "No passages found in document_chunk table with chunk_hash_id"
            )
            return

        logger.info(f"Processing {len(db_passages)} passages from database...")

        # Phase 1: Extract entities and sentences from all passages
        passages_data: List[Dict] = []
        all_entities: Set[str] = set()
        all_sentences: Set[str] = set()
        passage_entities_map: Dict[
            str, List[str]
        ] = {}  # passage_hash -> [entity_hashes]
        passage_sentences_map: Dict[
            str, List[str]
        ] = {}  # passage_hash -> [sentence_hashes]
        sentence_entities_map: Dict[
            str, List[str]
        ] = {}  # sentence_hash -> [entity_hashes]

        for row in db_passages:
            db_id, title, page_ref, content, passage_hash = row

            # Remove image references from passage text for processing
            passage_text = content
            if "[IMAGES]" in content:
                passage_text = content.split("[IMAGES]")[0].strip()

            # Skip passages that are too large (likely data quality issues)
            # Large passages cause memory issues and produce oversized sentences that exceed DB index limits
            if len(passage_text) > MAX_SPACY_LENGTH:
                logger.warning(
                    f"Skipping large passage ({len(passage_text)} chars) in {title} page {page_ref}. "
                    f"Exceeds MAX_SPACY_LENGTH ({MAX_SPACY_LENGTH}). Consider investigating upstream chunking."
                )
                self.stats["passages_skipped"] += 1
                continue

            passages_data.append(
                {
                    "db_id": db_id,
                    "title": title,
                    "page_ref": page_ref,
                    "content": passage_text,
                    "passage_hash": passage_hash,
                }
            )

            # Extract entities and sentences using spaCy
            doc = self.nlp(passage_text)
            # Extract entities (NER)
            entities = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
            # Extract sentences and filter out oversized ones (exceed PostgreSQL btree index limit)
            raw_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            sentences = []
            for sent in raw_sentences:
                sent_bytes = len(sent.encode('utf-8'))
                if sent_bytes > MAX_SENTENCE_BYTES:
                    logger.warning(
                        f"Filtering oversized sentence ({sent_bytes} bytes) from {title} page {page_ref}. "
                        f"Exceeds MAX_SENTENCE_BYTES ({MAX_SENTENCE_BYTES}). First 100 chars: {sent[:100]}..."
                    )
                    self.stats["sentences_filtered_oversized"] += 1
                else:
                    sentences.append(sent)

            entity_hashes = [compute_hash_id(ent, "entity-") for ent in entities]

            all_entities.update(entities)
            passage_entities_map[passage_hash] = entity_hashes

            sentence_hashes = [compute_hash_id(sent, "sentence-") for sent in sentences]

            all_sentences.update(sentences)
            passage_sentences_map[passage_hash] = sentence_hashes

            # Map sentences to entities (which entities appear in which sentences)
            for sent, sent_hash in zip(sentences, sentence_hashes):
                sent_doc = self.nlp(sent)
                sent_entities = [
                    ent.text.strip() for ent in sent_doc.ents if ent.text.strip()
                ]
                sent_entity_hashes = [
                    compute_hash_id(ent, "entity-") for ent in sent_entities
                ]
                sentence_entities_map[sent_hash] = sent_entity_hashes

            self.stats["passages_processed"] += 1
            self.stats["entities_extracted"] += len(entities)
            self.stats["sentences_extracted"] += len(sentences)

        self.stats["unique_entities"] = len(all_entities)
        self.stats["unique_sentences"] = len(all_sentences)

        logger.info(
            f"Extracted {self.stats['entities_extracted']} entities ({self.stats['unique_entities']} unique)"
        )
        logger.info(
            f"Extracted {self.stats['sentences_extracted']} sentences ({self.stats['unique_sentences']} unique)"
        )

        # Phase 2: Insert unique entities and sentences (triggers will handle embeddings)
        self._insert_entities(all_entities)
        self._insert_sentences(all_sentences)

        # Phase 3: Insert relationship mappings
        self._insert_passage_entity_mappings(passage_entities_map)
        self._insert_sentence_entity_mappings(sentence_entities_map)

        # Phase 4: Build graph structure (nodes + weighted edges)
        self._build_graph(
            passages_data,
            passage_entities_map,
            passage_sentences_map,
            sentence_entities_map,
        )

        logger.info(f"Created {self.stats['graph_nodes_created']} graph nodes")
        logger.info(f"Created {self.stats['graph_edges_created']} graph edges")

        # Log all statistics to MLflow
        for key, value in self.stats.items():
            mlflow.log_metric(key, value)

    def _insert_entities(self, entities: Set[str]) -> None:
        """Insert unique entities into lr_entity_embedding table."""
        if not entities:
            return

        entity_data = [
            (entity, compute_hash_id(entity, "entity-")) for entity in entities
        ]

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO lr_entity_embedding (entity_text, entity_hash_id)
                VALUES %s
                ON CONFLICT (entity_text) DO NOTHING
                """,
                entity_data,
                page_size=self.batch_size,
            )
        self.conn.commit()
        logger.info(f"Inserted {len(entities)} unique entities")

    def _insert_sentences(self, sentences: Set[str]) -> None:
        """Insert unique sentences into lr_sentence_embedding table."""
        if not sentences:
            return

        sentence_data = [
            (sentence, compute_hash_id(sentence, "sentence-")) for sentence in sentences
        ]

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO lr_sentence_embedding (sentence_text, sentence_hash_id)
                VALUES %s
                ON CONFLICT (sentence_text) DO NOTHING
                """,
                sentence_data,
                page_size=self.batch_size,
            )
        self.conn.commit()
        logger.info(f"Inserted {len(sentences)} unique sentences")

    def _insert_passage_entity_mappings(
        self, passage_entities: Dict[str, List[str]]
    ) -> None:
        """Insert passage-to-entity mappings into lr_passage_entity."""
        mappings = [
            (passage_hash, entity_hash)
            for passage_hash, entity_hashes in passage_entities.items()
            for entity_hash in entity_hashes
        ]

        if not mappings:
            return

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO lr_passage_entity (passage_hash_id, entity_hash_id)
                VALUES %s
                ON CONFLICT (passage_hash_id, entity_hash_id) DO NOTHING
                """,
                mappings,
                page_size=self.batch_size,
            )
        self.conn.commit()
        logger.info(f"Inserted {len(mappings)} passage-entity mappings")

    def _insert_sentence_entity_mappings(
        self, sentence_entities: Dict[str, List[str]]
    ) -> None:
        """Insert sentence-to-entity mappings into lr_sentence_entity."""
        mappings = [
            (sentence_hash, entity_hash)
            for sentence_hash, entity_hashes in sentence_entities.items()
            for entity_hash in entity_hashes
        ]

        if not mappings:
            return

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO lr_sentence_entity (sentence_hash_id, entity_hash_id)
                VALUES %s
                ON CONFLICT (sentence_hash_id, entity_hash_id) DO NOTHING
                """,
                mappings,
                page_size=self.batch_size,
            )
        self.conn.commit()
        logger.info(f"Inserted {len(mappings)} sentence-entity mappings")

    def _build_graph(
        self,
        passages_data: List[Dict],
        passage_entities: Dict[str, List[str]],
        passage_sentences: Dict[str, List[str]],
        sentence_entities: Dict[str, List[str]],
    ) -> None:
        """
        Build graph structure with nodes and weighted edges.

        Graph structure:
        - Nodes: passages, entities, sentences (each with type label)
        - Edges:
          1. passage → entity (weight: TF-IDF-like based on entity frequency)
          2. passage → sentence (weight: 1.0, structural containment)
          3. sentence → entity (weight: 1.0, structural containment)
          4. passage → passage (weight: 1.0, sequential adjacency)
        """
        # Collect all unique nodes
        all_nodes: Set[Tuple[str, str, str]] = set()  # (hash_id, type, text)

        # Add passage nodes
        for passage in passages_data:
            all_nodes.add(
                (
                    passage["passage_hash"],
                    "passage",
                    passage["content"][:500],  # Truncate for storage
                )
            )

        # Add entity nodes
        entity_texts: Dict[str, str] = {}  # hash -> text
        with self.conn.cursor() as cur:
            cur.execute("SELECT entity_hash_id, entity_text FROM lr_entity_embedding")
            for row in cur.fetchall():
                entity_hash, entity_text = row
                entity_texts[entity_hash] = entity_text
                all_nodes.add((entity_hash, "entity", entity_text))

        # Add sentence nodes
        sentence_texts: Dict[str, str] = {}  # hash -> text
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT sentence_hash_id, sentence_text FROM lr_sentence_embedding"
            )
            for row in cur.fetchall():
                sentence_hash, sentence_text = row
                sentence_texts[sentence_hash] = sentence_text
                all_nodes.add((sentence_hash, "sentence", sentence_text[:500]))

        # Insert nodes
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO lr_graph_node (node_hash_id, node_type, node_text)
                VALUES %s
                ON CONFLICT (node_hash_id) DO NOTHING
                """,
                list(all_nodes),
                page_size=self.batch_size,
            )
        self.conn.commit()
        self.stats["graph_nodes_created"] = len(all_nodes)

        # Build edges
        edges: List[Tuple[str, str, str, float]] = []  # (source, target, type, weight)

        # 1. Passage → Entity edges (with TF-IDF-like weights)
        # Count entity frequencies across all passages for IDF calculation
        entity_doc_freq: Counter[str] = Counter()  # entity_hash -> document frequency
        for entity_hashes in passage_entities.values():
            for entity_hash in set(entity_hashes):  # unique per passage
                entity_doc_freq[entity_hash] += 1

        num_passages = len(passages_data)

        for passage_hash, entity_hashes in passage_entities.items():
            # Count entity frequency in this passage (TF)
            entity_counts = Counter(entity_hashes)

            for entity_hash, tf in entity_counts.items():
                # TF-IDF-like weight: log(1 + tf) * log(num_docs / df)
                df = entity_doc_freq[entity_hash]
                idf = math.log(num_passages / df) if df > 0 else 0
                weight = math.log(1 + tf) * idf

                edges.append((passage_hash, entity_hash, "passage_entity", weight))

        # 2. Passage → Sentence edges (structural containment, weight 1.0)
        for passage_hash, sentence_hashes in passage_sentences.items():
            for sentence_hash in sentence_hashes:
                edges.append((passage_hash, sentence_hash, "passage_sentence", 1.0))

        # 3. Sentence → Entity edges (structural containment, weight 1.0)
        for sentence_hash, entity_hashes in sentence_entities.items():
            for entity_hash in entity_hashes:
                edges.append((sentence_hash, entity_hash, "sentence_entity", 1.0))

        # 4. Passage → Passage edges (sequential adjacency for same document)
        passages_by_doc: Dict[
            str, List[Tuple[int, str]]
        ] = {}  # title -> [(page_ref, hash)]
        for passage in passages_data:
            title = passage["title"]
            if title not in passages_by_doc:
                passages_by_doc[title] = []
            passages_by_doc[title].append(
                (passage["page_ref"], passage["passage_hash"])
            )

        for title, doc_passages in passages_by_doc.items():
            # Sort by page_ref
            doc_passages.sort(key=lambda x: x[0])
            # Create edges between adjacent passages
            for i in range(len(doc_passages) - 1):
                source_hash = doc_passages[i][1]
                target_hash = doc_passages[i + 1][1]
                edges.append((source_hash, target_hash, "passage_passage", 1.0))

        # Insert all edges
        if edges:
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO lr_graph_edge (source_hash_id, target_hash_id, edge_type, weight)
                    VALUES %s
                    ON CONFLICT (source_hash_id, target_hash_id, edge_type) DO NOTHING
                    """,
                    edges,
                    page_size=self.batch_size,
                )
            self.conn.commit()
            self.stats["graph_edges_created"] = len(edges)


def main(database_url: str, spacy_model: str, batch_size: int) -> None:
    """
    Main entry point for LinearRAG graph indexing stage.

    Reads passages from document_chunk table to build knowledge graph in Supabase lr_* tables.
    """
    logger.info(f"Loading spaCy model: {spacy_model}")
    # Disable parser to reduce memory usage, use rule-based sentence segmentation
    nlp = spacy.load(spacy_model, disable=["parser"])
    # Add simple rule-based sentencizer (lightweight alternative to parser)
    nlp.add_pipe("sentencizer")
    logger.info("SpaCy pipeline optimized: parser disabled, using sentencizer for sentence segmentation")

    logger.info(f"Connecting to database: {database_url}")
    conn = _connect(database_url)

    try:
        indexer = LinearRAGIndexer(conn, nlp, batch_size)
        indexer.process_chunks()
        logger.info("LinearRAG graph indexing completed successfully")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LinearRAG indexing stage: build knowledge graph from document_chunk table"
    )

    parser.add_argument(
        "--database_url", type=str, required=True, help="PostgreSQL database URL"
    )

    args = parser.parse_args()

    # Load parameters
    index_params = load_params("index")
    batch_size = int(index_params.get("batch_size", 32))
    spacy_model = str(index_params.get("spacy_model", "en_core_web_sm"))
    embedding_provider = str(index_params.get("embedding_provider", "ollama"))

    # MLflow tracking
    mlflow_connect()
    experiment_name = get_mlflow_experiment_name()
    mlflow.set_experiment(experiment_name)

    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("spacy_model", spacy_model)
    mlflow.log_param("embedding_provider", embedding_provider)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"linearrag_index_{timestamp}"

    # End any active run before starting new one (DVC context issue)
    mlflow.end_run()

    with mlflow.start_run(run_name=run_name):
        main(args.database_url, spacy_model, batch_size)
