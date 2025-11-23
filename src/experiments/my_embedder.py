# pip install ollama
import argparse
import asyncio

from openai import OpenAI
import ollama

# The IBM Granite Embedding 30M and 278M models models are text-only dense biencoder embedding models, with 30M available in English only and 278M serving multilingual use cases.
GRANITE_EMBEDDING = "granite-embedding:278m"

# Building upon the foundational models of the Qwen3 series, Qwen3 Embedding provides a comprehensive range of text embeddings models in various sizes
QWEN3_EMBEDDING = "qwen3-embedding:8b"

# State-of-the-art large embedding model from mixedbread.ai
MXBAI_EMBED_LARGE = "mxbai-embed-large:335m"

# A high-performing open embedding model with a large token context window.
NOMIC_EMBED_TEXT = "nomic-embed-text:137m-v1.5-fp16"


# Embedding model from BAAI mapping texts to vectors.
BGE_LARGE = "bge-large:335m"

# EmbeddingGemma is a 300M parameter embedding model from Google.
EMBEDDINGGEMMA = "embeddinggemma:300m"


# BGE-M3 is a new model from BAAI distinguished for its versatility in Multi-Functionality, Multi-Linguality, and Multi-Granularity.
BGE_M3 = "bge-m3:567m"

# Snowflake's frontier embedding model. Arctic Embed 2.0 adds multilingual support without sacrificing English performance or scalability.
SNOWFLAKE_ARCTIC_EMBED2 = "snowflake-arctic-embed2:568m"


openai_client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

# an enmpty model_name parameter will pull ALL available embedding models to ollama
def load_embedding_models(model_name: str = ""):
    model_list = ollama.list()
    model_names = []
    for m in model_list.models:
       print(m.model) 
       model_names.append(m.model)

    if (model_name == "" or model_name == GRANITE_EMBEDDING) and GRANITE_EMBEDDING not in model_names:
        print(f"ollama pulling {GRANITE_EMBEDDING}")
        ollama.pull(GRANITE_EMBEDDING)

    if (model_name == "" or model_name == QWEN3_EMBEDDING) and QWEN3_EMBEDDING not in model_names:
        print(f"ollama pulling {QWEN3_EMBEDDING}")
        ollama.pull(QWEN3_EMBEDDING)

    if (model_name == "" or model_name == MXBAI_EMBED_LARGE) and MXBAI_EMBED_LARGE not in model_names:
        print(f"ollama pulling {MXBAI_EMBED_LARGE}")
        ollama.pull(MXBAI_EMBED_LARGE)

    if (model_name == "" or model_name == NOMIC_EMBED_TEXT) and NOMIC_EMBED_TEXT not in model_names:
        print(f"ollama pulling {NOMIC_EMBED_TEXT}")
        ollama.pull(NOMIC_EMBED_TEXT)

    if (model_name == "" or model_name == BGE_LARGE) and BGE_LARGE not in model_names:
        print(f"ollama pulling {BGE_LARGE}")
        ollama.pull(BGE_LARGE)

    if (model_name == "" or model_name == EMBEDDINGGEMMA) and EMBEDDINGGEMMA not in model_names:
        print(f"ollama pulling {EMBEDDINGGEMMA}")
        ollama.pull(EMBEDDINGGEMMA)

    if (model_name == "" or model_name == BGE_M3) and BGE_M3 not in model_names:
        print(f"ollama pulling {BGE_M3}")
        ollama.pull(BGE_M3)

    if (model_name == "" or model_name == SNOWFLAKE_ARCTIC_EMBED2) and SNOWFLAKE_ARCTIC_EMBED2 not in model_names:
        print(f"ollama pulling {SNOWFLAKE_ARCTIC_EMBED2}")
        ollama.pull(SNOWFLAKE_ARCTIC_EMBED2)


def emb_text(text: str, model: str):

    # ATENCION: numbers with ollama have LESS precision! 
    # OPENAI_API: [-0.006824095733463764, 
    # OLLAMA_API: [-0.0068240957, 
    # Embedding with ollama
    # embedding_response = ollama.embed(model=model, input=input)
    # print(embedding_response.embeddings[0][:10])
    # Embed (batch)
    # ollama.embed(model=model, input=['The sky is blue because of rayleigh scattering', 'Grass is green because of chlorophyll'])

    return (
        openai_client.embeddings.create(input=text, model=model)
        .data[0]
        .embedding
    )




async def process_chunks(input: str, model: str):
    # print(model)

    # test embedding
    test_embedding = emb_text(input, model)
    embedding_dim = len(test_embedding)
    print(embedding_dim)
    print(test_embedding[:10])





async def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser(description="Run Docling ingestion on the bundled sample PDF.")

    parser.add_argument("--input", type=str, default="Hello World")
    parser.add_argument("--model", type=str, default=NOMIC_EMBED_TEXT)

    args = parser.parse_args()

    input = args.input
    model = args.model

    # check and pull the embedding model to ollama
    load_embedding_models(model_name = model)

    # run processing
    await process_chunks(input, model)

if __name__ == "__main__":
    # preload/pull all embedding models
    # load_embedding_models()

    asyncio.run(main())

