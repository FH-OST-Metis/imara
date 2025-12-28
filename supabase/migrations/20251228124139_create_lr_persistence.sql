-- Create tables for LinearRAG graph structure (prefixed with lr_)
-- Dual embedding system: Gemini (3072d) + Ollama (1024d)

-- Entity embeddings (extracted via NER from passages)
create table if not exists lr_entity_embedding (
  id integer primary key generated always as identity,
  entity_text text not null unique,
  entity_hash_id text not null unique,
  embedding_gemini halfvec(3072),
  embedding_ollama halfvec(1024),
  created_at timestamp with time zone default now()
);

create index on lr_entity_embedding using hnsw (embedding_gemini halfvec_cosine_ops);
create index on lr_entity_embedding using hnsw (embedding_ollama halfvec_cosine_ops);
create index on lr_entity_embedding (entity_hash_id);

-- Sentence embeddings (extracted from passages)
create table if not exists lr_sentence_embedding (
  id integer primary key generated always as identity,
  sentence_text text not null unique,
  sentence_hash_id text not null unique,
  embedding_gemini halfvec(3072),
  embedding_ollama halfvec(1024),
  created_at timestamp with time zone default now()
);

create index on lr_sentence_embedding using hnsw (embedding_gemini halfvec_cosine_ops);
create index on lr_sentence_embedding using hnsw (embedding_ollama halfvec_cosine_ops);
create index on lr_sentence_embedding (sentence_hash_id);

-- Passage to Entity mapping (many-to-many)
create table if not exists lr_passage_entity (
  id integer primary key generated always as identity,
  passage_hash_id text not null,
  entity_hash_id text not null,
  created_at timestamp with time zone default now(),
  unique (passage_hash_id, entity_hash_id)
);

create index on lr_passage_entity (passage_hash_id);
create index on lr_passage_entity (entity_hash_id);

-- Sentence to Entity mapping (many-to-many)
create table if not exists lr_sentence_entity (
  id integer primary key generated always as identity,
  sentence_hash_id text not null,
  entity_hash_id text not null,
  created_at timestamp with time zone default now(),
  unique (sentence_hash_id, entity_hash_id)
);

create index on lr_sentence_entity (sentence_hash_id);
create index on lr_sentence_entity (entity_hash_id);

-- Graph nodes (unified view of all nodes: passages, entities, sentences)
create table if not exists lr_graph_node (
  id integer primary key generated always as identity,
  node_hash_id text not null unique,
  node_type text not null check (node_type in ('passage', 'entity', 'sentence')),
  node_text text not null,
  created_at timestamp with time zone default now()
);

create index on lr_graph_node (node_hash_id);
create index on lr_graph_node (node_type);

-- Graph edges (connections between nodes with weights)
create table if not exists lr_graph_edge (
  id integer primary key generated always as identity,
  source_hash_id text not null,
  target_hash_id text not null,
  edge_type text not null,
  weight real not null default 1.0,
  created_at timestamp with time zone default now(),
  unique (source_hash_id, target_hash_id, edge_type)
);

create index on lr_graph_edge (source_hash_id);
create index on lr_graph_edge (target_hash_id);
create index on lr_graph_edge (edge_type);

-- Add hash_id column to document_chunk for graph integration
alter table document_chunk add column if not exists chunk_hash_id text;
create index if not exists idx_chunk_hash_id on document_chunk (chunk_hash_id);

-- Function to generate hash IDs from content (MD5-based, matching LinearRAG)
create or replace function util.compute_hash_id(content text, prefix text default '')
returns text
language plpgsql
immutable
as $$
begin
  return prefix || md5(content);
end;
$$;

-- Update existing chunks with hash IDs (if they don't have one)
update document_chunk
set chunk_hash_id = util.compute_hash_id(content, 'passage-')
where chunk_hash_id is null;

-- Content extraction functions for embedding triggers
create or replace function get_lr_entity_content(row_data lr_entity_embedding)
returns text 
language plpgsql 
immutable 
as $$
begin 
  return row_data.entity_text; 
end; 
$$;

create or replace function get_lr_sentence_content(row_data lr_sentence_embedding)
returns text 
language plpgsql 
immutable 
as $$
begin 
  return row_data.sentence_text; 
end; 
$$;

-- ========================================
-- GEMINI EMBEDDING TRIGGERS
-- ========================================

-- Trigger für lr_entity_embedding: Queue Gemini Embedding-Job bei INSERT
create trigger lr_embed_entity_gemini_on_insert
  after insert on lr_entity_embedding
  for each row
  when (NEW.embedding_gemini is null)
  execute function util.queue_embeddings('get_lr_entity_content', 'embedding_gemini');

-- Trigger für lr_sentence_embedding: Queue Gemini Embedding-Job bei INSERT
create trigger lr_embed_sentence_gemini_on_insert
  after insert on lr_sentence_embedding
  for each row
  when (NEW.embedding_gemini is null)
  execute function util.queue_embeddings('get_lr_sentence_content', 'embedding_gemini');

-- Clear Gemini embeddings when text changes (entity)
create trigger lr_clear_entity_embedding_gemini_on_update
  before update of entity_text
  on lr_entity_embedding
  for each row
  execute function util.clear_column('embedding_gemini');

-- Clear Gemini embeddings when text changes (sentence)
create trigger lr_clear_sentence_embedding_gemini_on_update
  before update of sentence_text
  on lr_sentence_embedding
  for each row
  execute function util.clear_column('embedding_gemini');

-- ========================================
-- OLLAMA EMBEDDING TRIGGERS
-- ========================================

-- Trigger für lr_entity_embedding: Queue Ollama Embedding-Job bei INSERT
create trigger lr_embed_entity_ollama_on_insert
  after insert on lr_entity_embedding
  for each row
  when (NEW.embedding_ollama is null)
  execute function util.queue_embeddings_ollama('get_lr_entity_content', 'embedding_ollama');

-- Trigger für lr_sentence_embedding: Queue Ollama Embedding-Job bei INSERT
create trigger lr_embed_sentence_ollama_on_insert
  after insert on lr_sentence_embedding
  for each row
  when (NEW.embedding_ollama is null)
  execute function util.queue_embeddings_ollama('get_lr_sentence_content', 'embedding_ollama');

-- Clear Ollama embeddings when text changes (entity)
create trigger lr_clear_entity_embedding_ollama_on_update
  before update of entity_text
  on lr_entity_embedding
  for each row
  execute function util.clear_column('embedding_ollama');

-- Clear Ollama embeddings when text changes (sentence)
create trigger lr_clear_sentence_embedding_ollama_on_update
  before update of sentence_text
  on lr_sentence_embedding
  for each row
  execute function util.clear_column('embedding_ollama');