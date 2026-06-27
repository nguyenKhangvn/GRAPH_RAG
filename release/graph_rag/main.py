import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# ----------------------------
# 1. Init Environment
from dotenv import load_dotenv
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

# 2. Imports Modules
from graph_rag.pipeline.graph_rag_pipeline import RAGPipeline
from graph_rag.modules.retrieval.embedding import LocalEmbeddingService
from graph_rag.config import PIPELINE_LLM_MODEL_NAME

def main():
    print(" Initializing GraphRAG System...")
    
    # Init Embedding Service (Load Model 1 lần duy nhất)
    try:
        embedder = LocalEmbeddingService()
    except (ValueError, TypeError) as e:
        print(" Cannot load Embedding Model. Check setup.")
        return

    # Init Pipeline
    pipeline = RAGPipeline(
        embedding_service=embedder,
        llm_model_name=PIPELINE_LLM_MODEL_NAME,
    )

    print("\n💬 Hệ thống đã sẵn sàng! Gõ 'exit' để thoát.")
    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        if not query.strip(): continue

        try:
            response = pipeline.run(query)
            print(f"\n AI: {response}")
        except (ValueError, TypeError) as e:
            print(f" Error: {e}")

    # Cleanup
    pipeline.close()

if __name__ == "__main__":
    main()