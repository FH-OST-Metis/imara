create or replace function get_chunk_content(row_data document_chunk)
returns text language plpgsql immutable as $$
begin return row_data.content; end; $$;

-- Trigger for TEXT
create trigger embed_text_on_insert
  after insert on document_chunk
  for each row
  execute function util.queue_embeddings('get_chunk_content', 'embedding_gemini');

