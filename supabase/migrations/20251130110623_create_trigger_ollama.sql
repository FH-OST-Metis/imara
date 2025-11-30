create trigger embed_ollama_text_on_insert
  after insert on document_chunk
  for each row
  execute function util.queue_embeddings_ollama('get_chunk_content', 'embedding_ollama');

-- Trigger to clear the embedding column on update
create trigger clear_document_embedding_ollama_on_update
  before update of title, page_ref, pic_ref, content -- must match the columns in embedding_input()
  on document_chunk
  for each row
  execute function util.clear_column('embedding_ollama');