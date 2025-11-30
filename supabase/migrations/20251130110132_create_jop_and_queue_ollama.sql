-- Queue for processing additional embedding jobs with Ollama
select pgmq.create('embedding_jobs_ollama');

CREATE OR REPLACE FUNCTION util.queue_embeddings_ollama()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  content_function text := TG_ARGV[0];
  embedding_column text := TG_ARGV[1];
BEGIN
  PERFORM pgmq.send(
    queue_name => 'embedding_jobs_ollama', -- HIER: Neue Queue
    msg => jsonb_build_object(
      'id', NEW.id,
      'schema', TG_TABLE_SCHEMA,
      'table', TG_TABLE_NAME,
      'contentFunction', content_function,
      'embeddingColumn', embedding_column
    )
  );
  RETURN NEW;
END;
$$;