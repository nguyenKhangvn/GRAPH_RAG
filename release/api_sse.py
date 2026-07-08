import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="GraphRAG Chatbot SSE API", 
    description="API Streaming SSE hỗ trợ Chat & Interactive Map", 
    version="2.0.0"
)

# ── 1. CẤU HÌNH CORS (Bắt buộc cho Frontend React gọi API Local) ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên sửa thành ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. ĐỊNH NGHĨA SCHEMAS (Request/Response Models) ───────────────────────────
class MessageRole(BaseModel):
    role: str      # "user" | "assistant"
    content: str   # Nội dung tin nhắn

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[MessageRole]] = []

# --- IMPORT VÀ KHỞI TẠO HỆ THỐNG GRAPHRAG THẬT ---
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(current_dir, "graph_rag", ".env"))
load_dotenv(os.path.join(current_dir, ".env"))

from graph_rag.pipeline.graph_rag_pipeline import RAGPipeline
from graph_rag.modules.retrieval.embedding import LocalEmbeddingService
from graph_rag.services.database import Neo4jService

print("📥 Khởi tạo Embedding service và RAG Pipeline...")
try:
    embedder = LocalEmbeddingService()
    # Let pipeline pick provider/key by model from environment.
    pipeline = RAGPipeline(embedding_service=embedder)
except Exception as e:
    print(f"❌ Lỗi khởi tạo Pipeline: {e}")
    sys.exit(1)


# ── 3. HÀM CORE ENGINE GRAPHRAG THẬT (Tạo Streaming Generator) ──────────────────
async def real_graphrag_engine(query: str, chat_history: List[MessageRole]):
    """
    Hàm logic xử lý của hệ thống lõi GraphRAG thực sự và trả ra Stream.
    Nhiệm vụ:
    1. Lấy dữ liệu trả về từ pipeline
    2. Streaming kết quả giả lập gõ chữ.
    3. Tạo toạ độ (Metadata fallback).
    """

    loop = asyncio.get_event_loop()
    try:
        start_time = time.time()
        
        # Lấy lịch sử dạng dict
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in chat_history] if chat_history else []
        
        result = await loop.run_in_executor(None, pipeline.run, query, history_dicts, "")
        answer = result["answer"]
        raw_meta = result["metadata"]
        print(f"✅ RAGPipeline hoàn thành trong {time.time() - start_time:.2f}s")
    except Exception as e:
         print(f"❌ Lỗi gọi RAGPipeline: {e}")
         answer = "Xin lỗi, hệ thống AI đang gặp sự cố khi truy vấn cơ sở dữ liệu."
         raw_meta = {}

    # 3. Tạo toạ độ dựa theo Database (Lấy Output Entity Node ra từ pipeline)
    current_intent = str(raw_meta.get("intent", "DISCOVERY"))
    
    locations = []
    source_nodes = (
        raw_meta.get("answered_route_nodes")
        or raw_meta.get("route_seed_nodes")
        or raw_meta.get("seed_nodes", [])
    )
    for node in source_nodes:
        attrs = node.get("attributes", {})
        
        # Thử lấy lat/lng từ các keyword thông dụng trong db (hỗ trợ cả cấp gốc hoặc attributes)
        lat = node.get("lat") or attrs.get("latitude") or attrs.get("lat")
        lng = node.get("lng") or attrs.get("longitude") or attrs.get("lng")
        
        # Nếu node đó thoả mãn có toạ độ -> Đẩy vào list pin
        if lat is not None and lng is not None:
            locations.append({
                "id": node.get("id"),
                "name": node.get("name") or attrs.get("name") or "Unknown Place",
                "type": node.get("labels", ["Place"])[0] if node.get("labels") else "Place",
                "coordinates": {"lat": float(lat), "lng": float(lng)}
            })
    
    # Nếu không tìm thấy node nào có toạ độ trọn vẹn mà câu hỏi có keyword tỉnh, 
    # fallback về mock hoặc bỏ trống (để FE không di chuyển)
    if not locations and "Tour" in current_intent:
        locations = [
            {"id": "fallback_1", "name": "Biển hồ T'Nưng", "coordinates": {"lat": 14.104908, "lng": 108.001920}},
             {"id": "fallback_2", "name": "Chùa Minh Thành", "coordinates": {"lat": 13.9749, "lng": 108.0029}}
        ]
        
    metadata = {
            "intent": current_intent,
            "locations": locations
        }

    # 4. Stream từng ký tự trả về
    chunk_size = 8
    for i in range(0, len(answer), chunk_size):
        chunk = answer[i:i + chunk_size]
        payload = json.dumps({"chunk": chunk}, ensure_ascii=False)
        yield f"event: message\ndata: {payload}\n\n"
        await asyncio.sleep(0.01) # gõ chữ với chunk bằng 8 cho tốc độ nhanh hơn
        
    # 5. Gửi Tọa độ cho Bản đồ ở cuối luồng
    meta_payload = json.dumps(metadata, ensure_ascii=False)
    yield f"event: metadata\ndata: {meta_payload}\n\n"
    
    # 6. End signal
    yield f"event: done\ndata: [DONE]\n\n"


# ── 4. ENDPOINT API QUAN TRỌNG NHẤT ───────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/chat")
async def chat_streaming_endpoint(request: ChatRequest):
    """
    Endpoint trả về luồng stream (StreamingResponse).
    Bắt buộc gán media_type="text/event-stream" chuẩn giao thức SSE.
    """
    if not request.query.strip():
        # Trả về 400 nếu bị rỗng, nhưng FastAPI StreamingResponse cần handle kỹ
        return {"error": "Câu hỏi không được để trống"}
        
    print(f"\n[RECEIVED] Query: {request.query}")
    print(f"[HISTORY] Found {len(request.chat_history)} previous messages.")

    # Khởi chạy generator (luồng thực thi của AI thật)
    generator_stream = real_graphrag_engine(request.query, request.chat_history)

    return StreamingResponse(
        generator_stream, 
        media_type="text/event-stream",
        # Các header bắt buộc cho SSE Stream không bị chặn hoặc nén bởi browser/proxy
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" # Ngăn proxy (Nginx) cache stream data
        }
    )

if __name__ == "__main__":
    import uvicorn
    # Mặc định chạy ở port 8000
    uvicorn.run("api_sse:app", host="0.0.0.0", port=8000, reload=True)
