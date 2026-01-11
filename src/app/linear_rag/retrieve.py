import argparse
from pathlib import Path
import sys
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import math
import os
from collections import defaultdict
from openai import OpenAI

# Ensure sibling 'utils' package is importable when running as script via path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds src/app

from utils.params_helper import load_params

import psycopg2
from psycopg2.extensions import connection as PGConnection

import spacy
from spacy.language import Language

import igraph as ig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _connect(database_url: str) -> PGConnection:
    """Connect to Postgres database."""
    return psycopg2.connect(database_url)


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1] range."""
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.ones_like(scores)
    return (scores - min_val) / (max_val - min_val)


class LinearRAGRetriever:
    """
    LinearRAG retrieval system using knowledge graph and entity-aware search.

    Implements the retrieval pipeline from LinearRAG:
    1. Extract seed entities from question using spaCy NER
    2. Expand entities through graph traversal with sentences
    3. Calculate passage scores combining DPR and entity bonuses
    4. Run Personalized PageRank for final ranking
    """

    def __init__(
        self,
        conn: PGConnection,
        nlp: Language,
        embedding_provider: str = "ollama",
        retrieval_config: Optional[Dict] = None,
    ):
        self.conn = conn
        self.nlp = nlp
        self.embedding_provider = embedding_provider

        # Retrieval configuration
        self.config = retrieval_config or {}
        self.retrieval_top_k = self.config.get("retrieval_top_k", 5)
        self.max_iterations = self.config.get("max_iterations", 3)
        self.iteration_threshold = self.config.get("iteration_threshold", 0.1)
        self.top_k_sentence = self.config.get("top_k_sentence", 3)
        self.passage_ratio = self.config.get("passage_ratio", 0.5)
        self.passage_node_weight = self.config.get("passage_node_weight", 1.0)
        self.damping = self.config.get("damping", 0.85)

        logger.info(
            f"Initializing LinearRAGRetriever with embedding_provider={embedding_provider}"
        )

        # Initialize embedding clients
        self._init_embedding_clients()

        # Load embeddings and graph structure
        self._load_embeddings()
        self._load_graph()
        self._build_mappings()

    def _init_embedding_clients(self):
        """Initialize embedding client(s) based on provider."""
        if self.embedding_provider == "ollama":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: uv add openai"
                )
            self.ollama_client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
                api_key="ollama",
            )
        elif self.embedding_provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai required. Install with: uv add google-generativeai"
                )
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_client = genai

    def _parse_halfvec(self, halfvec_str: str) -> np.ndarray:
        """Parse PostgreSQL halfvec string to numpy array."""
        if not halfvec_str:
            return np.array([])
        # halfvec is returned as "[0.1,0.2,...]"
        values = list(map(float, halfvec_str.strip("[]").split(",")))
        return np.array(values)

    def _format_vector_for_query(self, vector: np.ndarray) -> str:
        """Format numpy array as pgvector string for SQL queries."""
        return "[" + ",".join(map(str, vector.flatten())) + "]"

    def _load_embeddings(self):
        """Load entity, sentence, and passage metadata from database.
        Embeddings are queried via pgvector for better performance.
        Sentence embeddings are loaded lazily when needed.
        """
        embedding_col = f"embedding_{self.embedding_provider}"

        logger.info(f"Loading entity metadata...")
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT entity_hash_id, entity_text
                FROM lr_entity_embedding
                WHERE {embedding_col} IS NOT NULL
                ORDER BY entity_hash_id
            """)
            entity_rows = cur.fetchall()

        self.entity_hash_ids = [row[0] for row in entity_rows]
        self.entity_hash_id_to_text = {row[0]: row[1] for row in entity_rows}
        self.entity_hash_id_to_idx = {h: i for i, h in enumerate(self.entity_hash_ids)}

        logger.info(f"Loaded {len(self.entity_hash_ids)} entity records")

        logger.info(f"Loading sentence metadata...")
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT sentence_hash_id, sentence_text
                FROM lr_sentence_embedding
                WHERE {embedding_col} IS NOT NULL
                ORDER BY sentence_hash_id
            """)
            sentence_rows = cur.fetchall()

        self.sentence_hash_ids = [row[0] for row in sentence_rows]
        self.sentence_hash_id_to_text = {row[0]: row[1] for row in sentence_rows}
        self.sentence_hash_id_to_idx = {
            h: i for i, h in enumerate(self.sentence_hash_ids)
        }

        # Sentence embeddings loaded lazily
        self.sentence_embeddings = None
        self._sentence_embeddings_loaded = False

        logger.info(f"Loaded {len(self.sentence_hash_ids)} sentence records")

        logger.info(f"Loading passage metadata...")
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT dc.chunk_hash_id, dc.content
                FROM document_chunk dc
                WHERE dc.chunk_hash_id IS NOT NULL
                  AND dc.{embedding_col} IS NOT NULL
                ORDER BY dc.chunk_hash_id
            """)
            passage_rows = cur.fetchall()

        self.passage_hash_ids = [row[0] for row in passage_rows]
        self.passage_hash_id_to_text = {row[0]: row[1] for row in passage_rows}
        self.passage_hash_id_to_idx = {
            h: i for i, h in enumerate(self.passage_hash_ids)
        }

        logger.info(f"Loaded {len(self.passage_hash_ids)} passage records")

    def _load_graph(self):
        """Load graph structure from lr_graph_node and lr_graph_edge tables."""
        logger.info("Loading graph structure from database...")

        # Load nodes
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT node_hash_id, node_type, node_text
                FROM lr_graph_node
                ORDER BY node_hash_id
            """)
            node_rows = cur.fetchall()

        # Create igraph
        self.graph = ig.Graph(directed=False)
        node_hash_ids = [row[0] for row in node_rows]
        node_types = [row[1] for row in node_rows]
        node_texts = [row[2] for row in node_rows]

        # Add vertices
        self.graph.add_vertices(len(node_hash_ids))
        self.graph.vs["name"] = node_hash_ids
        self.graph.vs["type"] = node_types
        self.graph.vs["content"] = node_texts

        logger.info(f"Loaded {len(node_hash_ids)} graph nodes")

        # Load edges
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT source_hash_id, target_hash_id, edge_type, weight
                FROM lr_graph_edge
                ORDER BY source_hash_id, target_hash_id
            """)
            edge_rows = cur.fetchall()

        # Build edge list, filtering edges where both nodes exist
        node_hash_id_set = set(node_hash_ids)
        edges = []
        weights = []
        for row in edge_rows:
            source, target = row[0], row[1]
            if source in node_hash_id_set and target in node_hash_id_set:
                edges.append((source, target))
                weights.append(row[3])

        if len(edge_rows) > len(edges):
            logger.warning(
                f"Filtered {len(edge_rows) - len(edges)} edges with missing nodes"
            )

        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights

        logger.info(f"Loaded {len(edges)} graph edges")

    def _build_mappings(self):
        """Build mapping structures for efficient lookups."""
        logger.info("Building lookup mappings...")

        # Node name to vertex index
        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()
        }
        self.vertex_idx_to_node_name = {
            v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()
        }

        # Passage node indices
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id]
            for passage_id in self.passage_hash_ids
            if passage_id in self.node_name_to_vertex_idx
        ]

        # Entity to sentence mappings
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT sentence_hash_id, entity_hash_id
                FROM lr_sentence_entity
            """)
            rows = cur.fetchall()

        self.sentence_hash_id_to_entity_hash_ids = defaultdict(list)
        self.entity_hash_id_to_sentence_hash_ids = defaultdict(list)

        for sentence_hash_id, entity_hash_id in rows:
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id].append(
                entity_hash_id
            )
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id].append(
                sentence_hash_id
            )

        # Pre-compute lowercased text for efficient string matching
        logger.info("Pre-computing lowercased text cache...")
        self.entity_hash_id_to_text_lower = {
            h: text.lower() for h, text in self.entity_hash_id_to_text.items()
        }
        self.passage_hash_id_to_text_lower = {
            h: text.lower() for h, text in self.passage_hash_id_to_text.items()
        }

        logger.info("Mappings built successfully")

    def _ensure_sentence_embeddings_loaded(self):
        """Lazy-load sentence embeddings when first needed for entity expansion."""
        if self._sentence_embeddings_loaded:
            return

        logger.info("Loading sentence embeddings for entity expansion...")
        embedding_col = f"embedding_{self.embedding_provider}"

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT sentence_hash_id, {embedding_col}
                FROM lr_sentence_embedding
                WHERE {embedding_col} IS NOT NULL
                  AND sentence_hash_id = ANY(%s)
                ORDER BY sentence_hash_id
            """,
                (self.sentence_hash_ids,),
            )
            sentence_rows = cur.fetchall()

        # Pre-allocate array for better performance
        if sentence_rows:
            first_embedding = self._parse_halfvec(sentence_rows[0][1])
            embedding_dim = len(first_embedding)
            self.sentence_embeddings = np.empty(
                (len(sentence_rows), embedding_dim), dtype=np.float32
            )
            self.sentence_embeddings[0] = first_embedding

            for i in range(1, len(sentence_rows)):
                self.sentence_embeddings[i] = self._parse_halfvec(sentence_rows[i][1])
        else:
            self.sentence_embeddings = np.array([])

        self._sentence_embeddings_loaded = True
        logger.info(f"Loaded {len(sentence_rows)} sentence embeddings")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using Gemini or Ollama embedding API.
        Uses RETRIEVAL_QUERY task type for queries (vs RETRIEVAL_DOCUMENT for indexed data).
        """
        if self.embedding_provider == "gemini":
            return self._encode_gemini(text)
        elif self.embedding_provider == "ollama":
            return self._encode_ollama(text)
        else:
            raise ValueError(f"Unknown embedding_provider: {self.embedding_provider}")

    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts in batch for better performance.
        Falls back to sequential encoding if batch API not supported.
        """
        if self.embedding_provider == "gemini":
            return self._encode_gemini_batch(texts)
        elif self.embedding_provider == "ollama":
            return self._encode_ollama_batch(texts)
        else:
            raise ValueError(f"Unknown embedding_provider: {self.embedding_provider}")

    def _encode_gemini(self, text: str) -> np.ndarray:
        """Use Gemini API directly (same as edge function but with RETRIEVAL_QUERY task type)."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. Install with: uv add google-generativeai"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_query",  # Query vs document task type
            output_dimensionality=3072,
        )

        return np.array(result["embedding"])

    def _encode_gemini_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encode texts using Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. Install with: uv add google-generativeai"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        # Gemini embed_content supports lists of content
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=texts,
            task_type="retrieval_query",
            output_dimensionality=3072,
        )

        # Result contains list of embeddings
        return np.array([emb["values"] for emb in result["embedding"]])

    def _encode_ollama(self, text: str) -> np.ndarray:
        """Use Ollama API directly (same as edge function)."""
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # Required by client but unused by Ollama
        )

        response = client.embeddings.create(
            model=os.getenv("OLLAMA_MODEL", "bge-m3:567m"), input=text
        )

        return np.array(response.data[0].embedding)

    def _encode_ollama_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encode texts using Ollama API."""
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )

        # Ollama supports batch encoding via list input
        response = client.embeddings.create(
            model=os.getenv("OLLAMA_MODEL", "bge-m3:567m"), input=texts
        )

        return np.array([item.embedding for item in response.data])

    def retrieve(self, questions: List[Dict]) -> List[Dict]:
        """
        Retrieve relevant passages for given questions.

        Args:
            questions: List of dicts with 'question' and optional 'answer' keys

        Returns:
            List of dicts with retrieval results including ranked passages
        """
        retrieval_results = []

        for question_info in questions:
            question = question_info["question"]
            logger.info(f"Retrieving for question: {question[:100]}...")

            # Encode question
            question_embedding = self.encode_text(question)

            # Get seed entities from question
            (
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            ) = self.get_seed_entities(question, question_embedding)

            if len(seed_entities) > 0:
                logger.info(
                    f"Found {len(seed_entities)} seed entities: {seed_entities}"
                )
                sorted_passage_hash_ids, sorted_passage_scores = (
                    self.graph_search_with_seed_entities(
                        question_embedding,
                        seed_entity_indices,
                        seed_entities,
                        seed_entity_hash_ids,
                        seed_entity_scores,
                    )
                )
            else:
                logger.info("No seed entities found, using dense passage retrieval")
                sorted_passage_indices, sorted_passage_scores = (
                    self.dense_passage_retrieval(question_embedding)
                )
                sorted_passage_hash_ids = [
                    self.passage_hash_ids[idx] for idx in sorted_passage_indices
                ]

            # Get top-k passages
            final_passage_hash_ids = sorted_passage_hash_ids[: self.retrieval_top_k]
            final_passage_scores = sorted_passage_scores[: self.retrieval_top_k]
            final_passages = [
                self.passage_hash_id_to_text[h] for h in final_passage_hash_ids
            ]

            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "sorted_passage_hash_ids": final_passage_hash_ids,
                "gold_answer": question_info.get("answer", ""),
            }
            retrieval_results.append(result)

        return retrieval_results

    def get_seed_entities(
        self, question: str, question_embedding: np.ndarray
    ) -> Tuple[List[int], List[str], List[str], List[float]]:
        """Extract seed entities from question using spaCy NER and match to knowledge base using pgvector."""
        doc = self.nlp(question)
        question_entities = [ent.text.strip() for ent in doc.ents if ent.text.strip()]

        if len(question_entities) == 0:
            return [], [], [], []

        logger.info(
            f"Extracted {len(question_entities)} entities from question: {question_entities}"
        )

        # Encode question entities in batch for better performance
        question_entity_embeddings = self.encode_texts_batch(question_entities)

        # Similarity threshold to filter poor matches
        similarity_threshold = 0.5
        embedding_col = f"embedding_{self.embedding_provider}"

        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []

        # Use pgvector to find best matching entity for each question entity
        with self.conn.cursor() as cur:
            for query_entity_idx, (question_entity, entity_embedding) in enumerate(
                zip(question_entities, question_entity_embeddings)
            ):
                vector_str = self._format_vector_for_query(entity_embedding)

                # Use pgvector's cosine distance operator (<=>)
                # Note: <=> returns distance (0 = identical, 2 = opposite), so we convert to similarity
                cur.execute(
                    f"""
                    SELECT entity_hash_id, entity_text, 
                           1 - ({embedding_col} <=> %s::vector) / 2 as similarity
                    FROM lr_entity_embedding
                    WHERE {embedding_col} IS NOT NULL
                    ORDER BY {embedding_col} <=> %s::vector
                    LIMIT 1
                    """,
                    (vector_str, vector_str),
                )
                row = cur.fetchone()

                if row is None:
                    logger.warning(f"No entity match found for '{question_entity}'")
                    continue

                best_entity_hash_id, best_entity_text, best_entity_score = row

                # Skip if similarity is too low
                if best_entity_score < similarity_threshold:
                    logger.warning(
                        f"Low similarity ({best_entity_score:.3f}) for '{question_entity}', skipping"
                    )
                    continue

                # Skip if entity has no corresponding graph node
                if best_entity_hash_id not in self.node_name_to_vertex_idx:
                    logger.warning(
                        f"Entity '{best_entity_text}' ({best_entity_hash_id}) not found in graph, skipping"
                    )
                    continue

                # Get entity index from hash_id_to_idx mapping
                if best_entity_hash_id not in self.entity_hash_id_to_idx:
                    logger.warning(
                        f"Entity '{best_entity_text}' ({best_entity_hash_id}) not in entity index, skipping"
                    )
                    continue

                best_entity_idx = self.entity_hash_id_to_idx[best_entity_hash_id]
                seed_entity_indices.append(best_entity_idx)
                seed_entity_texts.append(best_entity_text)
                seed_entity_hash_ids.append(best_entity_hash_id)
                seed_entity_scores.append(float(best_entity_score))

        return (
            [int(i) for i in seed_entity_indices],
            seed_entity_texts,
            seed_entity_hash_ids,
            seed_entity_scores,
        )

    def graph_search_with_seed_entities(
        self,
        question_embedding: np.ndarray,
        seed_entity_indices: List[int],
        seed_entities: List[str],
        seed_entity_hash_ids: List[str],
        seed_entity_scores: List[float],
    ) -> Tuple[List[str], List[float]]:
        """Perform graph-based search starting from seed entities."""
        entity_weights, actived_entities = self.calculate_entity_scores(
            question_embedding,
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
        )
        passage_weights = self.calculate_passage_scores(
            question_embedding, actived_entities
        )
        node_weights = entity_weights + passage_weights

        ppr_sorted_passage_hash_ids, ppr_sorted_passage_scores = self.run_ppr(
            node_weights
        )
        return ppr_sorted_passage_hash_ids, ppr_sorted_passage_scores

    def calculate_entity_scores(
        self,
        question_embedding: np.ndarray,
        seed_entity_indices: List[int],
        seed_entities: List[str],
        seed_entity_hash_ids: List[str],
        seed_entity_scores: List[float],
    ) -> Tuple[np.ndarray, Dict]:
        """Calculate entity scores through iterative expansion via sentences."""
        # Ensure sentence embeddings are loaded for entity expansion
        self._ensure_sentence_embeddings_loaded()

        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))

        # Initialize with seed entities
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (
                seed_entity_idx,
                seed_entity_score,
                1,
            )
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score

        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1

        # Iterative expansion
        while len(current_entities) > 0 and iteration < self.max_iterations:
            new_entities = {}

            for entity_hash_id, (
                entity_id,
                entity_score,
                tier,
            ) in current_entities.items():
                if entity_score < self.iteration_threshold:
                    continue

                # Get sentences containing this entity
                sentence_hash_ids = [
                    sid
                    for sid in self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]
                    if sid not in used_sentence_hash_ids
                ]

                if not sentence_hash_ids:
                    continue

                # Calculate sentence similarities
                sentence_indices = [
                    self.sentence_hash_id_to_idx[sid] for sid in sentence_hash_ids
                ]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = (
                    question_embedding.reshape(-1, 1)
                    if len(question_embedding.shape) == 1
                    else question_embedding
                )
                sentence_similarities = np.dot(
                    sentence_embeddings, question_emb
                ).flatten()

                # Get top-k sentences
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][
                    : self.top_k_sentence
                ]

                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)

                    # Get entities in this sentence
                    entity_hash_ids_in_sentence = (
                        self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    )

                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score

                        if next_entity_score < self.iteration_threshold:
                            continue

                        # Skip if entity has no corresponding graph node
                        if next_entity_hash_id not in self.node_name_to_vertex_idx:
                            continue

                        next_entity_node_idx = self.node_name_to_vertex_idx[
                            next_entity_hash_id
                        ]
                        entity_weights[next_entity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (
                            next_entity_node_idx,
                            next_entity_score,
                            iteration + 1,
                        )

            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1

        logger.info(
            f"Entity expansion completed: {len(actived_entities)} entities activated"
        )
        return entity_weights, actived_entities

    def calculate_passage_scores(
        self, question_embedding: np.ndarray, actived_entities: Dict
    ) -> np.ndarray:
        """Calculate passage scores combining DPR and entity bonuses."""
        passage_weights = np.zeros(len(self.graph.vs["name"]))

        # Dense passage retrieval scores
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(
            question_embedding
        )
        dpr_passage_scores_list: List[float] = min_max_normalize(
            np.array(dpr_passage_scores)
        ).tolist()

        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores_list[i]
            passage_text_lower = self.passage_hash_id_to_text_lower[passage_hash_id]

            # Calculate entity bonuses
            for entity_hash_id, (
                entity_id,
                entity_score,
                tier,
            ) in actived_entities.items():
                entity_lower = self.entity_hash_id_to_text_lower[entity_hash_id]
                entity_occurrences = passage_text_lower.count(entity_lower)

                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = (
                        entity_score * math.log(1 + entity_occurrences) / denom
                    )
                    total_entity_bonus += entity_bonus

            # Combined score
            passage_score = self.passage_ratio * dpr_passage_score + math.log(
                1 + total_entity_bonus
            )
            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.passage_node_weight

        return passage_weights

    def run_ppr(self, node_weights: np.ndarray) -> Tuple[List[str], List[float]]:
        """Run Personalized PageRank with node weights as reset probabilities."""
        reset_prob = np.where(
            np.isnan(node_weights) | (node_weights < 0), 0, node_weights
        )

        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )

        # Extract passage scores
        doc_scores = np.array(
            [pagerank_scores[idx] for idx in self.passage_node_indices]
        )
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]

        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]]
            for i in sorted_indices_in_doc_scores
        ]

        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def dense_passage_retrieval(
        self, question_embedding: np.ndarray, top_k: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """Dense passage retrieval using pgvector cosine similarity search."""
        if top_k is None:
            top_k = len(self.passage_hash_ids)  # Return all for backward compatibility

        embedding_col = f"embedding_{self.embedding_provider}"
        vector_str = self._format_vector_for_query(question_embedding)

        # Use pgvector's cosine distance operator for fast similarity search
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_hash_id, 
                       1 - ({embedding_col} <=> %s::vector) / 2 as similarity
                FROM document_chunk
                WHERE {embedding_col} IS NOT NULL
                  AND chunk_hash_id = ANY(%s)
                ORDER BY {embedding_col} <=> %s::vector
                LIMIT %s
                """,
                (vector_str, self.passage_hash_ids, vector_str, top_k),
            )
            rows = cur.fetchall()

        # Build result matching expected format
        passage_hash_id_to_rank = {row[0]: idx for idx, row in enumerate(rows)}
        sorted_passage_indices = [
            self.passage_hash_id_to_idx[row[0]]
            for row in rows
            if row[0] in self.passage_hash_id_to_idx
        ]
        sorted_passage_scores = [
            float(row[1]) for row in rows if row[0] in self.passage_hash_id_to_idx
        ]

        return sorted_passage_indices, sorted_passage_scores


def main(
    database_url: str,
    spacy_model: str,
    embedding_provider: str,
    retrieval_config: Dict,
    question: str,
):
    """
    Main entry point for LinearRAG retrieval.

    Demonstrates retrieval for a single question.
    """
    logger.info(f"Loading spaCy model: {spacy_model}")
    nlp = spacy.load(spacy_model)

    logger.info(f"Connecting to database: {database_url}")
    conn = _connect(database_url)

    try:
        retriever = LinearRAGRetriever(conn, nlp, embedding_provider, retrieval_config)

        questions = [{"question": question, "answer": ""}]
        results = retriever.retrieve(questions)

        logger.info("=" * 80)
        logger.info(f"Question: {question}")
        logger.info("-" * 80)
        for i, (passage, score) in enumerate(
            zip(results[0]["sorted_passage"], results[0]["sorted_passage_scores"]), 1
        ):
            logger.info(f"\nRank {i} (score={score:.4f}):")
            logger.info(f"{passage[:200]}...")
        logger.info("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LinearRAG retrieval: graph-based passage ranking"
    )

    parser.add_argument(
        "--database_url", type=str, required=True, help="PostgreSQL database URL"
    )
    parser.add_argument(
        "--question", type=str, required=True, help="Question to retrieve passages for"
    )

    args = parser.parse_args()

    # Load parameters
    index_params = load_params("index")
    retrieval_params = load_params("retrieval")

    spacy_model = str(index_params.get("spacy_model", "en_core_web_sm"))
    embedding_provider = str(index_params.get("embedding_provider", "ollama"))

    main(
        args.database_url,
        spacy_model,
        embedding_provider,
        retrieval_params,
        args.question,
    )
