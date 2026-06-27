import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from graph_rag.services.database import Neo4jService

# ── Config riêng cho bge-m3 ─────────────────────────────────────────────────
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_DIM = 1024
BGE_M3_SUFFIX = "_bge_m3"
BGE_M3_EMBEDDING_PROP = "embedding_bge_m3"
BATCH_SIZE = 200  # bge-m3 nặng hơn, giảm batch

NODE_CONFIG = {
    "TouristAttraction": {
        "text_fn": lambda r: f"{r['name']} {r.get('description','') or ''}".strip(),
        "vector_index":   f"tourist_vec_idx{BGE_M3_SUFFIX}",
        "fulltext_index": "tourist_ft_idx",
    },
    "Restaurant": {
        "text_fn": lambda r: f"{r['name']} {r.get('type','') or ''} {r.get('address','') or ''}".strip(),
        "vector_index":   f"restaurant_vec_idx{BGE_M3_SUFFIX}",
        "fulltext_index": "restaurant_ft_idx",
    },
    "Accommodation": {
        "text_fn": lambda r: f"{r['name']} {r.get('type','') or ''} {r.get('address','') or ''}".strip(),
        "vector_index":   f"accommodation_vec_idx{BGE_M3_SUFFIX}",
        "fulltext_index": "accommodation_ft_idx",
    },
    "Tour": {
        "text_fn": lambda r: f"{r['name']} {r.get('description','') or ''}".strip(),
        "vector_index":   f"tour_vec_idx{BGE_M3_SUFFIX}",
        "fulltext_index": "tour_ft_idx",
    },
    "Event": {
        "text_fn": lambda r: f"{r['name']} {r.get('category','') or ''}".strip(),
        "vector_index":   f"event_vec_idx{BGE_M3_SUFFIX}",
        "fulltext_index": "event_ft_idx",
    },
}


def load_bge_m3():
    """Load bge-m3 model."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {BGE_M3_MODEL}...")
    model = SentenceTransformer(BGE_M3_MODEL)
    print(f"  ✅ Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    return model


def compute_and_store_embeddings(driver, model, label: str, config: dict):
    """Tính embedding bge-m3 và lưu vào property riêng."""
    with driver.session() as session:
        nodes = session.run(
            f"MATCH (n:{label}) RETURN n.id AS id, "
            + ", ".join([f"n.{p} AS {p}" for p in
                         ["name", "description", "type", "address", "category"]])
        ).data()

    if not nodes:
        print(f"  ⚠  [{label}] Không có node nào.")
        return 0

    print(f"  [{label}] {len(nodes)} nodes → tính bge-m3 embedding...", flush=True)

    texts = [config["text_fn"](n) or n.get("name", "") for n in nodes]
    ids   = [n["id"] for n in nodes]

    updated = 0
    for start in range(0, len(nodes), BATCH_SIZE):
        batch_ids   = ids[start:start + BATCH_SIZE]
        batch_texts = texts[start:start + BATCH_SIZE]

        vectors = model.encode(batch_texts, show_progress_bar=False).tolist()

        with driver.session() as session:
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (n {{id: row.id}})
                SET n.{BGE_M3_EMBEDDING_PROP} = row.vec
                """,
                rows=[{"id": i, "vec": v} for i, v in zip(batch_ids, vectors)]
            )
        updated += len(batch_ids)
        print(f"    {updated}/{len(nodes)} updated...", end="\r", flush=True)

    print(f"  ✅ [{label}] {updated} bge-m3 embeddings stored.      ")
    return updated


def create_vector_index(session, label: str, index_name: str, dim: int, prop: str):
    """Tạo vector index trên property riêng."""
    try:
        session.run(f"""
            CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
            FOR (n:{label}) ON (n.{prop})
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)
        print(f"  ✅ Vector index '{index_name}' created (dim={dim}, prop={prop}).")
    except (ValueError, TypeError) as e:
        print(f"  ⚠  Vector index '{index_name}' error: {e}")


def main():
    print("═" * 65)
    print("  GraphRAG bge-m3 Index Setup")
    print("  Model: BAAI/bge-m3 (1024 dim)")
    print("  Embedding property: embedding_bge_m3")
    print("═" * 65)

    driver = Neo4jService.get_driver()

    # 1. Load model
    print("\n[1/3] Loading bge-m3 model...")
    model = load_bge_m3()

    # 2. Compute & store embeddings
    print("\n[2/3] Computing & storing bge-m3 embeddings...")
    t0 = time.time()
    total = 0
    for label, cfg in NODE_CONFIG.items():
        total += compute_and_store_embeddings(driver, model, label, cfg)
    print(f"\n  Total: {total} bge-m3 embeddings stored in {time.time()-t0:.1f}s")

    # 3. Create vector indexes
    print("\n[3/3] Creating bge-m3 vector indexes...")
    vector_names = []
    with driver.session() as session:
        for label, cfg in NODE_CONFIG.items():
            if cfg["vector_index"]:
                create_vector_index(session, label, cfg["vector_index"], BGE_M3_DIM, BGE_M3_EMBEDDING_PROP)
                vector_names.append(cfg["vector_index"])

    # 4. Verify
    print("\n  Verifying indexes...")
    with driver.session() as session:
        rows = session.run(
            "SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR' RETURN name, type, state"
        ).data()
        for r in rows:
            status = "✅" if r["state"] == "ONLINE" else "⏳ POPULATING"
            print(f"  {status} {r['name']}")

    Neo4jService.close_driver()

    # 5. Print .env config
    print("\n" + "═" * 65)
    print("  DONE! Thêm vào graph_rag/.env để chuyển sang bge-m3:")
    print("═" * 65)
    print(f"\n  EMBEDDING_MODEL_NAME={BGE_M3_MODEL}")
    print(f"  EMBEDDING_DIMENSION={BGE_M3_DIM}")
    print(f"  VECTOR_INDEXES_STR={','.join(vector_names)}")
    print(f"\n  Để quay lại MiniLM, xóa/comment 3 dòng trên.")
    print("═" * 65)


if __name__ == "__main__":
    main()
