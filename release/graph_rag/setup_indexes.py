"""
graph_rag/setup_indexes.py
═══════════════════════════════════════════════════════════════════════════════
Tạo embedding cho tất cả nodes và tạo Vector + Fulltext indexes trong Neo4j.

Chạy 1 lần:
    cd e:\\DACK\\craw
    .venv\\Scripts\\python.exe graph_rag/setup_indexes.py
═══════════════════════════════════════════════════════════════════════════════
"""
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from graph_rag.services.database import Neo4jService
from graph_rag.modules.retrieval.embedding import LocalEmbeddingService
from graph_rag.config import EMBEDDING_DIMENSION

# ── Cấu hình embedding cho từng node label ────────────────────────────────────
# text_fn : hàm lấy text từ record để tạo embedding
# vector_index_name : tên index sẽ tạo
# fulltext_index_name: tên fulltext index (None = không tạo)
NODE_CONFIG = {
    "TouristAttraction": {
        "text_fn": lambda r: f"{r['name']} {r.get('description','') or ''}".strip(),
        "vector_index":   "tourist_vec_idx",
        "fulltext_index": "tourist_ft_idx",
        "ft_props":       ["name"],
    },
    "Restaurant": {
        "text_fn": lambda r: f"{r['name']} {r.get('type','') or ''} {r.get('address','') or ''}".strip(),
        "vector_index":   "restaurant_vec_idx",
        "fulltext_index": "restaurant_ft_idx",
        "ft_props":       ["name"],
    },
    "Accommodation": {
        "text_fn": lambda r: f"{r['name']} {r.get('type','') or ''} {r.get('address','') or ''}".strip(),
        "vector_index":   "accommodation_vec_idx",
        "fulltext_index": "accommodation_ft_idx",
        "ft_props":       ["name"],
    },
    "Tour": {
        "text_fn": lambda r: f"{r['name']} {r.get('description','') or ''}".strip(),
        "vector_index":   "tour_vec_idx",
        "fulltext_index": "tour_ft_idx",
        "ft_props":       ["name"],
    },
    "Event": {
        "text_fn": lambda r: f"{r['name']} {r.get('category','') or ''}".strip(),
        "vector_index":   "event_vec_idx",
        "fulltext_index": "event_ft_idx",
        "ft_props":       ["name"],
    },
    "Dish": {
        "text_fn": lambda r: f"{r['name']} {r.get('description','') or ''}".strip(),
        "vector_index":   "dish_vec_idx",  # Thêm vector index cho Dish
        "fulltext_index": "dish_ft_idx",
        "ft_props":       ["name", "category", "description"],
    },
    "TravelAgency": {
        "text_fn": lambda r: f"{r['name']} {r.get('address','') or ''}".strip(),
        "vector_index":   None,
        "fulltext_index": "agency_ft_idx",
        "ft_props":       ["name"],
    },
}

BATCH_SIZE = 500


def compute_and_store_embeddings(driver, embedder, label: str, config: dict):
    """Tính embedding và lưu vào thuộc tính `embedding` của node."""
    with driver.session() as session:
        nodes = session.run(
            f"MATCH (n:{label}) RETURN n.id AS id, "
            + ", ".join([f"n.{p} AS {p}" for p in
                         ["name", "description", "type", "address", "category"]])
        ).data()

    if not nodes:
        print(f"  ⚠  [{label}] Không có node nào.")
        return 0

    print(f"  [{label}] {len(nodes)} nodes → tính embedding...", flush=True)

    texts = [config["text_fn"](n) or n.get("name", "") for n in nodes]
    ids   = [n["id"] for n in nodes]

    updated = 0
    for start in range(0, len(nodes), BATCH_SIZE):
        batch_ids   = ids[start:start + BATCH_SIZE]
        batch_texts = texts[start:start + BATCH_SIZE]

        # Batch embed toàn bộ cùng lúc
        vectors = embedder.embed_texts(batch_texts)

        # Batch write to Neo4j
        with driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (n {id: row.id})
                SET n.embedding = row.vec
                """,
                rows=[{"id": i, "vec": v} for i, v in zip(batch_ids, vectors)]
            )
        updated += len(batch_ids)
        print(f"    {updated}/{len(nodes)} updated...", end="\r", flush=True)

    print(f"  ✅ [{label}] {updated} embeddings stored.      ")
    return updated


def create_vector_index(session, label: str, index_name: str, dim: int):
    """Tạo vector index (bỏ qua nếu đã có)."""
    try:
        session.run(f"""
            CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
            FOR (n:{label}) ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)
        print(f"  ✅ Vector index '{index_name}' created (or already exists).")
    except (ValueError, TypeError) as e:
        print(f"  ⚠  Vector index '{index_name}' error: {e}")


def create_fulltext_index(session, label: str, index_name: str, props: list):
    """Tạo fulltext index (bỏ qua nếu đã có)."""
    props_str = ", ".join([f"n.{p}" for p in props])
    try:
        session.run(f"""
            CREATE FULLTEXT INDEX `{index_name}` IF NOT EXISTS
            FOR (n:{label}) ON EACH [{props_str}]
        """)
        print(f"  ✅ Fulltext index '{index_name}' created (or already exists).")
    except (Neo4jClientError, ServiceUnavailable) as e:
        print(f"  ⚠  Fulltext index '{index_name}' error: {e}")


def update_config(vector_map: dict, fulltext_map: dict):
    """Cập nhật VECTOR_INDEXES và FULLTEXT_INDEXES trong config.py."""
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Build new lists
    v_list = [f'    "{v}"' for v in sorted(set(vector_map.values())) if v]
    ft_list = [f'    "{v}"' for v in sorted(set(fulltext_map.values())) if v]

    import re
    content = re.sub(
        r'VECTOR_INDEXES\s*=\s*\[.*?\]',
        'VECTOR_INDEXES = [\n' + ',\n'.join(v_list) + '\n]',
        content, flags=re.DOTALL
    )
    content = re.sub(
        r'FULLTEXT_INDEXES\s*=\s*\[.*?\]',
        'FULLTEXT_INDEXES = [\n' + ',\n'.join(ft_list) + '\n]',
        content, flags=re.DOTALL
    )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  ✅ config.py updated with new index names.")


def main():
    print("═" * 65)
    print("  GraphRAG Index Setup")
    print("═" * 65)

    # ── 1. Kết nối ──────────────────────────────────────────────────
    driver = Neo4jService.get_driver()

    # ── 2. Load embedding model ─────────────────────────────────────
    print("\n[1/3] Loading embedding model...")
    embedder = LocalEmbeddingService()

    # ── 3. Compute và store embeddings ──────────────────────────────
    print("\n[2/3] Computing & storing embeddings...")
    t0 = time.time()
    total = 0
    for label, cfg in NODE_CONFIG.items():
        total += compute_and_store_embeddings(driver, embedder, label, cfg)
    print(f"\n  Total: {total} embeddings stored in {time.time()-t0:.1f}s")

    # ── 4. Create indexes ────────────────────────────────────────────
    print("\n[3/3] Creating indexes in Neo4j...")
    vector_map  = {}
    fulltext_map = {}

    with driver.session() as session:
        for label, cfg in NODE_CONFIG.items():
            if cfg["vector_index"]:
                create_vector_index(session, label, cfg["vector_index"], EMBEDDING_DIMENSION)
                vector_map[label] = cfg["vector_index"]
            if cfg["fulltext_index"]:
                create_fulltext_index(session, label, cfg["fulltext_index"], cfg["ft_props"])
                fulltext_map[label] = cfg["fulltext_index"]

    # ── 5. Update config.py ──────────────────────────────────────────
    update_config(vector_map, fulltext_map)

    # ── 6. Verify ────────────────────────────────────────────────────
    print("\n  Verifying indexes...")
    with driver.session() as session:
        rows = session.run(
            "SHOW INDEXES YIELD name, type, state WHERE type IN ['VECTOR','FULLTEXT'] RETURN name, type, state"
        ).data()
        for r in rows:
            status = "✅" if r["state"] == "ONLINE" else "⏳ POPULATING"
            print(f"  {status} [{r['type']:8}] {r['name']}")

    Neo4jService.close_driver()
    print("\n═" * 65)
    print("  Setup complete! Run eval_e2e.py to evaluate the system.")
    print("═" * 65)


if __name__ == "__main__":
    main()
