# Ollama Embeddings Setup

This directory contains the Docker Compose configuration for running Ollama with the BGE-M3 embedding model.

## Table of Contents

- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Installing NVIDIA Container Toolkit (Linux)](#installing-nvidia-container-toolkit-linux)
  - [Installing NVIDIA Container Toolkit (Windows/WSL2)](#installing-nvidia-container-toolkit-windowswsl2)
  - [Setting Up Apple Silicon GPU Acceleration (MPS)](#setting-up-apple-silicon-gpu-acceleration-mps)
- [Quick Start](#quick-start)
- [GPU Acceleration](#gpu-acceleration)
  - [Enabling NVIDIA GPU (Linux/Windows)](#enabling-nvidia-gpu-linuxwindows)
  - [Enabling Apple Silicon GPU (M1/M2/M3/M4 Macs)](#enabling-apple-silicon-gpu-m1m2m3m4-macs)
  - [CPU-Only Mode](#cpu-only-mode)
- [Network Configuration for Supabase Edge Functions](#network-configuration-for-supabase-edge-functions)
- [Useful Commands](#useful-commands)
- [Model Information](#model-information)
- [Troubleshooting](#troubleshooting)
  - [Port 11434 Already in Use](#port-11434-already-in-use)
  - [DNS Resolution Failures](#dns-resolution-failures)
  - [GPU Not Detected](#gpu-not-detected)
  - [Model Not Found](#model-not-found)
  - [Connection Refused from Edge Functions](#connection-refused-from-edge-functions)
- [Additional Models](#additional-models)

## Prerequisites

Before starting Ollama, ensure you have Docker and Docker Compose installed. If you want GPU acceleration, follow the appropriate setup instructions below.

### System Requirements

**Minimum Requirements (CPU-only):**
- Docker Engine 20.10+
- Docker Compose v2.0+
- 8GB RAM
- 10GB disk space for models

**GPU Requirements:**
- **NVIDIA GPU**: Compute capability 7.0+ (RTX 2000 series or newer), NVIDIA drivers 525.x+
- **Apple Silicon**: M1/M2/M3/M4 Mac with macOS 13.3+

### Installing NVIDIA Container Toolkit (Linux)

If you have an NVIDIA GPU and want to use GPU acceleration on Linux:

#### 1. Verify NVIDIA Drivers

First, ensure NVIDIA drivers are installed:

```bash
nvidia-smi
```

You should see output showing your GPU information. If not, install NVIDIA drivers first.

#### 2. Add NVIDIA Container Toolkit Repository

```bash
# Add GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository to apt sources
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### 3. Install NVIDIA Container Toolkit

```bash
# Update package list
sudo apt-get update

# Install toolkit
sudo apt-get install -y nvidia-container-toolkit
```

#### 4. Configure Docker

```bash
# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker
```

#### 5. Verify Installation

Test that Docker can access your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU information displayed. If successful, you can now use GPU acceleration with Ollama.

### Installing NVIDIA Container Toolkit (Windows/WSL2)

For Windows users with NVIDIA GPUs:

#### 1. Prerequisites

- Windows 10/11 with WSL2 enabled
- Ubuntu 20.04+ running in WSL2
- NVIDIA GPU drivers for Windows (version 525.x or higher)
  - Download from: https://www.nvidia.com/Download/index.aspx
- Docker Desktop for Windows with WSL2 backend enabled

#### 2. Verify GPU Access in WSL2

Open your WSL2 terminal and verify the GPU is accessible:

```bash
nvidia-smi
```

If this works, your Windows NVIDIA drivers are properly configured for WSL2.

#### 3. Install NVIDIA Container Toolkit in WSL2

Follow the same steps as Linux installation above (steps 2-5) inside your WSL2 Ubuntu environment.

#### 4. Configure Docker Desktop

In Docker Desktop:
1. Go to **Settings** → **Resources** → **WSL Integration**
2. Enable integration with your WSL2 Ubuntu distribution
3. Click **Apply & Restart**

#### 5. Verify Installation

From your WSL2 terminal:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Setting Up Apple Silicon GPU Acceleration (MPS)

For Apple Silicon Macs (M1/M2/M3/M4):

#### 1. Prerequisites

- macOS 13.3 (Ventura) or later
- Docker Desktop for Mac 4.25.0 or later
- Apple Silicon Mac (M1/M2/M3/M4)

#### 2. Enable GPU Support in Docker Desktop

1. Open **Docker Desktop**
2. Go to **Settings** → **Features in development**
3. Enable **Use Rosetta for x86/amd64 emulation on Apple Silicon** (if available)
4. Enable **VirtioFS** for better performance
5. Click **Apply & Restart**

#### 3. Verify MPS Support

Currently, Docker Desktop on Apple Silicon has experimental GPU support. The MPS (Metal Performance Shaders) backend is enabled by default for Apple Silicon.

**Note:** As of Docker Desktop 4.x, MPS support for containers is limited. For best performance with Ollama on Apple Silicon, consider:
- Running Ollama natively (outside Docker) for full Metal acceleration
- Using the Docker container for portability with CPU-only mode

#### 4. Check Ollama Performance

After starting Ollama (see Quick Start below), monitor resource usage:

```bash
# Check if Ollama is using GPU resources
docker stats ollama-embeddings
```

For optimal performance on Apple Silicon, native Ollama installation is recommended:
```bash
brew install ollama
ollama serve
```

## Quick Start

### 1. Start Ollama

```bash
cd ollama
docker compose up -d
```

This will:
- Start the Ollama service on port `11434`
- Create a persistent volume for model storage at `./models`
- Enable health checks to ensure the service is running

### 2. Pull the BGE-M3 Model

Once Ollama is running, pull the BGE-M3 embedding model:

```bash
docker exec ollama-embeddings ollama pull bge-m3:567m
```

This downloads the BGE-M3 model (567M parameters) which provides:
- **1024-dimensional embeddings** (vs OpenAI's 1536)
- Multi-functionality (dense, lexical, multi-vector retrieval)
- Multi-lingual support
- Multi-granularity (sentence, passage, document level)

### 3. Verify the Model

Check that the model was successfully pulled:

```bash
docker exec ollama-embeddings ollama list
```

You should see `bge-m3:567m` in the list of available models.

### 4. Test the API

Test the embedding API directly:

```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "bge-m3:567m",
  "prompt": "Hello World"
}'
```

You should receive a JSON response with a 1024-dimensional embedding vector.

## GPU Acceleration

**Before enabling GPU acceleration**, ensure you have completed the appropriate Prerequisites installation above for your platform.

### Enabling NVIDIA GPU (Linux/Windows)

After installing NVIDIA Container Toolkit (see [Prerequisites](#installing-nvidia-container-toolkit-linux) or [WSL2 setup](#installing-nvidia-container-toolkit-windowswsl2)), the GPU configuration is already enabled in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

If it's commented out, uncomment these lines, then restart:

```bash
docker compose down
docker compose up -d
```

Verify GPU is being used:

```bash
# Check Ollama logs for GPU detection
docker logs ollama-embeddings

# Monitor GPU usage while running embeddings
nvidia-smi -l 1
```

### Enabling Apple Silicon GPU (M1/M2/M3/M4 Macs)

**Note:** MPS (Metal Performance Shaders) support in Docker is experimental. For best performance on Apple Silicon, consider running Ollama natively outside Docker.

To enable MPS in Docker, uncomment the MPS section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: mps
          count: all
```

Then restart the service:

```bash
docker compose down
docker compose up -d
```

**Alternative (Recommended for Apple Silicon):**

Run Ollama natively for full Metal acceleration:

```bash
brew install ollama
ollama serve
```

Then update your `.env` file:
```bash
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### CPU-Only Mode

By default, Ollama runs on CPU if no GPU configuration is set. No additional configuration is needed. 

**Performance Note:** BGE-M3 (567M parameters) will be significantly slower on CPU:
- GPU: ~50-200ms per embedding
- CPU: ~500-2000ms per embedding

## Network Configuration for Supabase Edge Functions

When accessing Ollama from Supabase Edge Functions running in Docker (via `supabase start`), you need to use `host.docker.internal` instead of `localhost`:

- **From host machine**: `http://localhost:11434/v1`
- **From Supabase Edge Functions**: `http://host.docker.internal:11434/v1`

This is configured in your `.env` file:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
OLLAMA_MODEL=bge-m3:567m
OLLAMA_API_KEY=ollama  # Required by OpenAI client, but unused
```

## Useful Commands

### View Ollama Logs
```bash
docker logs ollama-embeddings -f
```

### Stop Ollama
```bash
docker compose down
```

### Restart Ollama
```bash
docker compose restart
```

### Remove Ollama and Models
```bash
docker compose down -v
```

## Model Information

**BGE-M3** (BAAI General Embedding M3)
- **Parameters**: 567M
- **Embedding Dimensions**: 1024
- **Context Length**: 8192 tokens
- **Languages**: 100+ languages
- **Use Cases**: Semantic search, retrieval, classification, clustering

## Troubleshooting

### Port 11434 Already in Use

If you see the error `failed to bind host port 0.0.0.0:11434/tcp: address already in use`, another Ollama instance is running.

**Check what's using the port:**

```bash
sudo lsof -i :11434
```

**Solution 1: Stop the system Ollama service (if running)**

```bash
# Stop the system service
sudo systemctl stop ollama

# Disable it from starting on boot (optional)
sudo systemctl disable ollama

# Now start the Docker container
cd ollama && docker compose up -d
```

**Solution 2: Use the existing Ollama service**

If you prefer to use the system Ollama instead of Docker:

```bash
# Remove Docker setup
cd ollama && docker compose down
cd .. && rm -rf ollama/

# Pull the model with system Ollama
ollama pull bge-m3:567m

# Update .env to use localhost
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### DNS Resolution Failures

If you see errors like `dial tcp: lookup registry.ollama.ai: read: connection refused` or `pull model manifest` failures, the container cannot resolve DNS.

**Symptoms:**
```
Error: pull model manifest: Get "https://registry.ollama.ai/...": dial tcp: lookup registry.ollama.ai on 127.0.0.53:53: read udp 127.0.0.1:xxxxx->127.0.0.53:53: read: connection refused
```

**Solution:**

The DNS configuration is already included in `docker-compose.yml`:

```yaml
dns:
  - 8.8.8.8
  - 8.8.4.4
```

If you still have DNS issues:

1. Verify the DNS configuration is in your docker-compose.yml
2. Restart the container: `docker compose down && docker compose up -d`
3. Test DNS resolution: `docker exec ollama-embeddings nslookup registry.ollama.ai`

### GPU Not Detected

If GPU is not being used despite correct configuration:

**For NVIDIA GPUs:**

```bash
# Verify NVIDIA Container Toolkit is installed
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Check Docker runtime configuration
docker info | grep -i runtime

# View Ollama container logs for GPU detection
docker logs ollama-embeddings | grep -i gpu
```

**Common issues:**
- NVIDIA Container Toolkit not installed → See [Prerequisites](#installing-nvidia-container-toolkit-linux)
- Docker not configured for NVIDIA runtime → Run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`
- GPU configuration commented out in docker-compose.yml → Uncomment the deploy section

**For Apple Silicon:**

MPS support in Docker is limited. Consider running Ollama natively:
```bash
brew install ollama
ollama serve
```

### Model Not Found

If you get "model not found" errors, ensure the model is pulled:

```bash
docker exec ollama-embeddings ollama list
docker exec ollama-embeddings ollama pull bge-m3:567m
```

### Connection Refused from Edge Functions

If Supabase Edge Functions can't connect to Ollama:

1. Verify Ollama is running: `docker ps | grep ollama`
2. Check the health status: `docker inspect ollama-embeddings | grep -A 5 Health`
3. Test from host: `curl http://localhost:11434/api/tags`
4. Ensure `OLLAMA_BASE_URL` uses `host.docker.internal` in your `.env`

## Additional Models

You can pull additional embedding models:

```bash
# Smaller, faster models
docker exec ollama-embeddings ollama pull nomic-embed-text:137m-v1.5-fp16

# Larger models
docker exec ollama-embeddings ollama pull snowflake-arctic-embed2:568m
```

See your Python experiments at `src/experiments/my_embedder.py` for more model options.
