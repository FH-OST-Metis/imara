ALTER TABLE document_chunk 
ADD COLUMN embedding_ollama halfvec(1024);

CREATE INDEX ON document_chunk 
USING hnsw (embedding_ollama halfvec_cosine_ops);