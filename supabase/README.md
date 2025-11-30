# Supabase Local Development Environment

Supabase provides a local development environment that allows developers to easily manage databases, authentication, and storage. The local setup enables testing and development of backend features without connecting to external servers. With the Supabase CLI, projects can be quickly initialized and configured. Changes can be tested locally and later deployed to production, supporting an efficient workflow for modern web applications.

This setup includes all core features, such as authentication, databases, and storage buckets. Everything runs inside Docker containers, so Docker must be installed on your system.

## Dual Embedding Architecture

This project uses two embedding systems in parallel to generate vector embeddings for semantic search:

### Gemini Embedding (Cloud)
- **Provider:** Google Gemini AI (cloud-based)
- **Dimensions:** 3072
- **Model:** `text-embedding-004`
- **Cost:** Paid API (per-request charges)
- **Use Case:** Production deployments, high-quality embeddings
- **Setup:** Requires `GEMINI_API_KEY` in `.env`

### Ollama Embedding (Local)
- **Provider:** Local Ollama server with BGE-M3 model
- **Dimensions:** 1024
- **Model:** `bge-m3:567m` (BAAI General Embedding)
- **Cost:** Free (local compute, optional GPU acceleration)
- **Use Case:** Development, cost-effective alternative, offline work
- **Setup:** Requires Ollama running locally - see [Ollama Setup Guide](../ollama/README.md)

### How It Works

When a document is inserted into the `document_chunk` table:
1. Two database triggers fire automatically
2. Jobs are queued in separate message queues (`embedding_jobs` for Gemini, `embedding_jobs_ollama` for Ollama)
3. Cron jobs process the queues every 10 seconds
4. Edge Functions (`embed` and `embed_ollama`) generate embeddings
5. Embeddings are stored in `embedding_gemini` and `embedding_ollama` columns

Both systems run independently - you can use one or both depending on your needs.

## Prerequisites

Before starting, ensure you have:

- **Docker Desktop** - Running and accessible (required for all Supabase services)
- **Supabase CLI** - Installed ([installation guide](https://supabase.com/docs/guides/local-development/cli/getting-started))
- **Environment File** - Copy `.env-template` to `.env` and configure:
  - `GEMINI_API_KEY` - **Required** for Gemini embeddings
  - `OLLAMA_BASE_URL` - Optional (default: `http://host.docker.internal:11434/v1`)
  - `OLLAMA_MODEL` - Optional (default: `bge-m3:567m`)
- **Ollama Server** (optional) - For local embeddings, see [Ollama Setup Guide](../ollama/README.md)

## Serving Project Local

After you have set up local Supabase ENV (Install and Link). **See install instructions below**

> **Notice:** Docker has to be installed and up and running

> **Notice:** For Windows: Supabase CLI is installed by Scoop: https://scoop.sh. Don't forget to reboot Windows.

> **Notice:** For Windows: Has to be executed in PowerShell

### Start Supabase

```bash
supabase start
```

This starts all Supabase services in Docker containers.

### Serve Edge Functions

You have three options for serving Edge Functions:

#### Option 1: Serve All Functions (Recommended)

```bash
supabase functions serve --env-file .env --no-verify-jwt
```

This serves both `embed` (Gemini) and `embed_ollama` (Ollama) functions simultaneously.

#### Option 2: Serve Individual Functions

If you need to serve functions in separate terminals for debugging:

```bash
# Terminal 1 - Gemini embedding
supabase functions serve embed --env-file .env --no-verify-jwt

# Terminal 2 - Ollama embedding (requires Ollama running)
supabase functions serve embed_ollama --env-file .env --no-verify-jwt
```

#### Option 3: Serve Without Ollama

If you only want Gemini embeddings (no local Ollama):

```bash
supabase functions serve embed --env-file .env --no-verify-jwt
```

**What is `--env-file .env`?**  
Loads environment variables (API keys, Ollama config) from your `.env` file into the Edge Functions runtime.

**What is `--no-verify-jwt`?**  
Disables JWT verification for local development, allowing you to test functions without authentication tokens.

### Verify Ollama is Running (Optional)

If using local Ollama embeddings, verify it's accessible:

```bash
curl http://localhost:11434/api/tags
```

You should see a list of available models including `bge-m3:567m`.

## Environment Variables

Create a `.env` file in the project root (copy from `.env-template`):

### Required Variables

```bash
# Gemini Embedding (cloud)
GEMINI_API_KEY=your_gemini_api_key_here  # Get from: https://aistudio.google.com/app/apikey
```

### Optional Variables (Ollama)

```bash
# Ollama Embedding (local) - Only needed if using Ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1  # Default: connects to local Ollama
OLLAMA_MODEL=bge-m3:567m                               # Default: BGE-M3 model (1024 dimensions)
OLLAMA_API_KEY=ollama                                  # Default: placeholder (unused by Ollama)
```

### Other Variables

```bash
# Database connection (auto-provided by Supabase)
SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres

# MLflow and S3 (if using experiment tracking)
MLFLOW_BACKEND_STORE_URI=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

## Verification & Testing

### Check Supabase Status

```bash
supabase status
```

Shows all running services, ports, and connection URLs.

### Check Queue Status

Connect to the database and check message queues:

```bash
psql postgresql://postgres:postgres@127.0.0.1:54322/postgres
```

```sql
-- Check Gemini queue metrics
SELECT * FROM pgmq.metrics('embedding_jobs');

-- Check Ollama queue metrics
SELECT * FROM pgmq.metrics('embedding_jobs_ollama');

-- View active cron jobs
SELECT * FROM cron.job;
```

### View Function Logs

```bash
# Gemini embedding function logs
supabase functions logs embed

# Ollama embedding function logs
supabase functions logs embed_ollama
```

### Test Embedding Generation

Insert a test document and verify both embeddings are generated:

```sql
-- Insert test document
INSERT INTO document_chunk (title, page_ref, pic_ref, content)
VALUES ('Test Document', 1, NULL, 'This is a test document for embedding generation.');

-- Check if embeddings were generated (wait ~15 seconds for cron jobs)
SELECT 
  id,
  title,
  embedding_gemini IS NOT NULL as has_gemini_embedding,
  embedding_ollama IS NOT NULL as has_ollama_embedding
FROM document_chunk
ORDER BY created_at DESC
LIMIT 1;
```

Both `has_gemini_embedding` and `has_ollama_embedding` should be `true`.

## Important Supabase CLI Commands

The Supabase CLI offers essential commands for managing local and remote development environments. Here are the key commands for environment and migration management:

### Install Supabase CLI

https://supabase.com/docs/guides/local-development/cli/getting-started?queryGroups=platform&platform=macos


### Initialization and Project Linking

> **Notice:** This step has already been completed and committed. **Do not run it again.**

```bash
supabase init # switched to beta, due a bug on the stable version of the cli
```
Initializes a new Supabase project in the current directory.

> **Notice:** The linking has to be done on each local device. **Do this with first time connecting new local ENV **

```bash
supabase login # switched to beta, due a bug on the stable version of the cli
supabase link --project-ref <PROJECT_ID> # switched to beta, due a bug on the stable version of the cli
```
Links the local repository to an existing Supabase project.

### Synchronize Schema Changes

```bash
supabase db pull # switched to beta, due a bug on the stable version of the cli
```
Pulls schema changes from the dashboard and creates a new migration.

### Start Local Development

```bash
supabase start # switched to beta, due a bug on the stable version of the cli
```
Starts the local Supabase environment (Docker required).

### Create and Apply Migrations

**Manual migration:**
```bash
supabase migration new <migration_name> # switched to beta, due a bug on the stable version of the cli
```
Creates a new empty migration file.

**Migration from SQL script:**
```bash
supabase migration new <migration_name> < script.sql # switched to beta, due a bug on the stable version of the cli
```
Creates a migration directly from an existing SQL script.

**Apply migrations:**
```bash
supabase db reset # switched to beta, due a bug on the stable version of the cli
```
Resets the local database and applies all migrations.

### Automatic Schema Diff Migration

```bash
supabase db diff -f <migration_name> # switched to beta, due a bug on the stable version of the cli
```
Generates a migration based on changes in the local database.

---

These commands enable efficient management of development, staging, and production environments with Supabase and GitHub Actions.

## Troubleshooting

### Windows: Port Access Permissions

If you see this error:

```bash
failed to start docker container: Error response from daemon: ports are not available: exposing port TCP 0.0.0.0:54322 -> 127.0.0.1:0: listen tcp 0.0.0.0:54322: bind: An attempt was made to access a socket in a way forbidden by its access permissions.
Try rerunning the command with --debug to troubleshoot the error.
```

Execute this in PowerShell with Admin Permissions:

```bash
net stop winnat
docker start container_name
net start winnat
```

### Ollama Connection Failed

**Symptoms:**
- `embed_ollama` function shows errors in logs
- Queue `embedding_jobs_ollama` fills up but jobs fail
- Error: "Connection refused" or "host.docker.internal unreachable"

**Solutions:**

1. **Verify Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Check Ollama container status:**
   ```bash
   cd ollama && docker ps | grep ollama-embeddings
   ```

3. **Start Ollama if not running:**
   ```bash
   cd ollama && docker compose up -d
   ```

4. **Verify BGE-M3 model is pulled:**
   ```bash
   docker exec ollama-embeddings ollama list
   ```
   
   If missing, pull it:
   ```bash
   docker exec ollama-embeddings ollama pull bge-m3:567m
   ```

5. **Check DNS configuration:** See [Ollama Troubleshooting Guide](../ollama/README.md#troubleshooting) for DNS resolution issues.

### Embeddings Not Generated

**Symptoms:**
- Inserted documents don't have embeddings after 30+ seconds
- `embedding_gemini` or `embedding_ollama` columns remain NULL

**Solutions:**

1. **Check if Edge Functions are running:**
   ```bash
   # Should show both embed and embed_ollama
   curl http://127.0.0.1:54321/functions/v1/
   ```

2. **Verify cron jobs are scheduled:**
   ```bash
   psql postgresql://postgres:postgres@127.0.0.1:54322/postgres -c "SELECT * FROM cron.job;"
   ```
   
   Should show `process-embeddings` and `process-embeddings-ollama` jobs.

3. **Check queue for pending jobs:**
   ```bash
   psql postgresql://postgres:postgres@127.0.0.1:54322/postgres
   ```
   ```sql
   SELECT * FROM pgmq.metrics('embedding_jobs');
   SELECT * FROM pgmq.metrics('embedding_jobs_ollama');
   ```

4. **View function logs for errors:**
   ```bash
   supabase functions logs embed --tail 50
   supabase functions logs embed_ollama --tail 50
   ```

5. **Manually trigger processing:**
   ```sql
   SELECT util.process_embeddings();
   SELECT util.process_embeddings_ollama();
   ```

### Missing API Key

**Symptoms:**
- Gemini embeddings fail with "API key not found" or "401 Unauthorized"

**Solution:**

1. Verify `.env` file exists and contains `GEMINI_API_KEY`:
   ```bash
   grep GEMINI_API_KEY .env
   ```

2. Restart functions with `--env-file` flag:
   ```bash
   supabase functions serve --env-file .env --no-verify-jwt
   ```

3. Get API key from: https://aistudio.google.com/app/apikey

### Queue Growing Indefinitely

**Symptoms:**
- Queue metrics show increasing `queue_length` but no `total_messages` processed
- Database growing in size

**Solutions:**

1. **Check for stuck jobs:**
   ```sql
   SELECT msg_id, enqueued_at, vt, read_ct 
   FROM pgmq.q_embedding_jobs 
   ORDER BY enqueued_at DESC LIMIT 10;
   ```

2. **Clear stuck jobs (use with caution):**
   ```sql
   -- Archive old jobs
   SELECT pgmq.archive('embedding_jobs', msg_id) 
   FROM pgmq.q_embedding_jobs 
   WHERE vt < now() - interval '1 hour';
   ```

3. **Restart Edge Functions** to reset worker state.

### GPU Not Used by Ollama

If Ollama is running slowly despite having a GPU, see the [Ollama GPU Setup Guide](../ollama/README.md#prerequisites) for NVIDIA Container Toolkit installation instructions.