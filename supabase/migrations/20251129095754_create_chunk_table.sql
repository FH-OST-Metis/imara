-- Table to store documents with embeddings
create table document_chunk (
  id integer primary key generated always as identity,
  title text not null,
  page_ref integer not null,
  pic_ref text,
  content text not null,
  -- Dimension count has been looked up for embedding. 3072 fro gemini-embedding-001
  embedding_gemini halfvec(3072),
  created_at timestamp with time zone default now()
);

-- Index for vector search over document embeddings
create index on document_chunk using hnsw (embedding_gemini halfvec_cosine_ops);