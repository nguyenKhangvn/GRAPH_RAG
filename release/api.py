import sys
import os
import time
import json
import asyncio
import logging
import hashlib
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import queue
import threading
import socket
from collections import OrderedDict, defaultdict, deque

# Monkey-patch socket.getaddrinfo to force IPv4 and prevent Errno -5 DNS resolution issues on Render
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family == socket.AF_UNSPEC or family == 0:
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)
socket.getaddrinfo = _ipv4_only_getaddrinfo

from datetime import date
from typing import List, AsyncGenerator, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dotenv import load_dotenv
GRAPH_RAG_DOTENV_PATH = os.path.join(current_dir, "graph_rag", ".env")
ROOT_DOTENV_PATH = os.path.join(current_dir, ".env")
load_dotenv(ROOT_DOTENV_PATH, override=True)
load_dotenv(GRAPH_RAG_DOTENV_PATH, override=False)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from graph_rag.pipeline.graph_rag_pipeline import RAGPipeline
from graph_rag.modules.retrieval.embedding import LocalEmbeddingService


if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

# Suppress noisy Neo4j notification warnings (deprecation, unrecognized property)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

APP_LOGGER = logging.getLogger("backend.api")

app = FastAPI(title="GraphRAG Chatbot API", description="API Cho Trợ lý Du Lịch", version="2.0.0")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

embedder = None
pipeline = None

APP_ENV = os.getenv("APP_ENV", os.getenv("ENV", "dev")).lower()
INCLUDE_DEBUG_METADATA = APP_ENV in {"dev", "staging", "local", "test"}
ROUTE_MONITOR_LOGGER = logging.getLogger("monitoring.route_optimizer")
ROUTE_MONITOR_LOGGER.setLevel(logging.INFO)

MAPBOX_INTERNAL_DAILY_LIMIT = int(os.getenv("MAPBOX_INTERNAL_DAILY_LIMIT", "500"))
MAPBOX_RATE_LIMIT_PER_MINUTE = int(os.getenv("MAPBOX_RATE_LIMIT_PER_MINUTE", "10"))
MAPBOX_CACHE_TTL_SECONDS = int(os.getenv("MAPBOX_CACHE_TTL_SECONDS", "3600"))

CORS_ORIGINS = [
    o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000").split(",")
    if o.strip()
]
ROUTE_CACHE_MAX_SIZE = int(os.getenv("ROUTE_CACHE_MAX_SIZE", "500"))
NOMINATIM_CACHE_MAX_SIZE = int(os.getenv("NOMINATIM_CACHE_MAX_SIZE", "1000"))

_mapbox_daily_usage_date = date.today().isoformat()
_mapbox_daily_usage_count = 0
_mapbox_rate_windows = defaultdict(deque)
_mapbox_route_cache: OrderedDict = OrderedDict()
_mapbox_guard_lock = asyncio.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global embedder, pipeline
    if pipeline is not None:
        APP_LOGGER.info("rag_pipeline_already_initialized")
        return

    APP_LOGGER.info("initializing_rag_pipeline")
    try:
        embedder = LocalEmbeddingService()
        # Let LLMService resolve provider/key from env based on active model.
        pipeline = RAGPipeline(embedding_service=embedder)
    except Exception:
        APP_LOGGER.exception("rag_pipeline_init_failed")
        raise

    # Pre-warm BGE cross-encoder model to avoid cold-start timeout on first request
    try:
        from graph_rag.config import BGE_CANDIDATE_SCORING_MODEL
        from graph_rag.modules.context.bge_scorer import _load_model
        APP_LOGGER.info("pre_warming_bge_model model=%s", BGE_CANDIDATE_SCORING_MODEL)
        _load_model(BGE_CANDIDATE_SCORING_MODEL)
        APP_LOGGER.info("bge_model_ready")
    except Exception:
        APP_LOGGER.warning("bge_model_prewarm_failed", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    global pipeline
    if pipeline is None:
        return
    try:
        pipeline.close()
    except Exception:
        APP_LOGGER.exception("rag_pipeline_close_failed")


# --- SCHEMAS ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Message] = []
    current_location: str = ""


class RoutePoint(BaseModel):
    lat: float
    lng: float


class DirectionsRequest(BaseModel):
    points: List[RoutePoint]
    profile: str = "driving"
    geometries: str = "geojson"
    overview: str = "full"


def _normalize_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _build_route_cache_key(payload: DirectionsRequest) -> str:
    coords_key = ";".join([f"{p.lng:.6f},{p.lat:.6f}" for p in payload.points])
    return f"{payload.profile}|{payload.overview}|{payload.geometries}|{coords_key}"


def _prune_expired_route_cache(now_ts: float) -> None:
    expired_keys = [
        key
        for key, value in _mapbox_route_cache.items()
        if value.get("expires_at", 0) <= now_ts
    ]
    for key in expired_keys:
        _mapbox_route_cache.pop(key, None)


def _normalize_profile(profile: str) -> str:
    allowed = {"driving", "walking", "cycling"}
    profile_norm = str(profile or "driving").lower().strip()
    return profile_norm if profile_norm in allowed else "driving"


def _resolve_mapbox_token() -> str:
    # Reload .env here so token updates are picked up without waiting for a manual server restart.
    load_dotenv(GRAPH_RAG_DOTENV_PATH, override=True)
    load_dotenv(ROOT_DOTENV_PATH, override=True)
    return (
        os.getenv("MAPBOX_ACCESS_TOKEN")
        or os.getenv("MAPBOX_SERVER_ACCESS_TOKEN")
        or os.getenv("VITE_MAPBOX_ACCESS_TOKEN")
        or ""
    )


def _build_tour_plan_safety(metadata: dict) -> dict:
    daily_cluster_plan = metadata.get("daily_cluster_plan") or []
    route_metrics = metadata.get("route_optimizer_metrics") or {}

    unsafe_days = []
    has_rule_flags = False
    for day in daily_cluster_plan:
        if not isinstance(day, dict):
            continue
        if "rule_ok" not in day:
            continue
        has_rule_flags = True
        if day.get("rule_ok") is False:
            unsafe_days.append(day.get("day"))

    safe_for_fe = not bool(unsafe_days)
    reason = "ok"
    warning = ""

    if has_rule_flags and unsafe_days:
        reason = "daily_rule_violation"
        warning = (
            "Lịch trình vẫn được hiển thị, nhưng một số chặng có thể đã vượt quãng đường đề xuất."

        )

    return {
        "safe_for_fe": safe_for_fe,
        "reason": reason,
        "warning": warning,
        "unsafe_days": unsafe_days,
        "max_hop_km_config": route_metrics.get("max_hop_km_config"),
        "max_hop_km_actual": route_metrics.get("max_hop_km_actual"),
        "total_distance_km": route_metrics.get("total_distance_km"),
    }


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    s = unicodedata.normalize("NFD", str(value).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(s.split())


def _is_single_place_lookup(query: str) -> bool:
    q = _normalize_text(query)
    signals = [
        "o dau",
        "nam o dau",
        "dia chi",
        "cho nao",
        "vi tri",
    ]
    return any(s in q for s in signals)


def _filter_focus_locations(query: str, metadata: dict, locations: list) -> list:
    if not locations:
        return locations

    intent = str(metadata.get("intent", "")).upper()
    if intent in {"TOUR_PLAN", "DISTANCE_QUERY"}:
        return locations

    if not _is_single_place_lookup(query):
        return locations

    entities = metadata.get("entities") or []
    focus_names = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        e_type = str(e.get("type") or "").lower()
        if e_type in {"location", "duration", "time"}:
            continue
        e_name = str(e.get("name") or "").strip()
        if e_name:
            focus_names.append(_normalize_text(e_name))

    if focus_names:
        focused = []
        for loc in locations:
            loc_name = _normalize_text(loc.get("name") or "")
            if any(fn in loc_name or loc_name in fn for fn in focus_names):
                focused.append(loc)
        if focused:
            return focused

    return locations


def _filter_by_answer_mentions(answer: str, locations: list) -> list:
    if not answer or not locations:
        return locations
    norm_answer = _normalize_text(answer)
    matched = []
    for loc in locations:
        name = loc.get("name") or ""
        norm_name = _normalize_text(name)
        if not norm_name:
            continue
        if norm_name in norm_answer:
            matched.append(loc)
            continue
        prefixes = ["chua ", "thac ", "nha hang ", "khach san ", "homestay ", "resort ", "lang du lich ", "lang ", "bai bien ", "bien ", "eo ", "cu lao ", "hon "]
        clean_name = norm_name
        for pfx in prefixes:
            if norm_name.startswith(pfx):
                clean_name = norm_name[len(pfx):]
                break
        if len(clean_name) >= 3 and clean_name in norm_answer:
            matched.append(loc)
    return matched if matched else locations


# --- SSE GENERATOR ---
async def sse_generator(query: str, chat_history: list, current_location: str = "", user_gps: str = "") -> AsyncGenerator[str, None]:
    try:
        if pipeline is None:
            raise RuntimeError("RAG pipeline is not initialized")

        token_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        streamed_any = False

        def on_token(token: str):
            """Called from pipeline thread for each LLM token."""
            loop.call_soon_threadsafe(token_queue.put_nowait, ("token", token))

        def run_pipeline():
            """Runs in a background thread."""
            try:
                result = pipeline.run(
                    query,
                    chat_history=chat_history,
                    current_location=current_location,
                    user_gps=user_gps,
                    on_token=on_token,
                )
                loop.call_soon_threadsafe(token_queue.put_nowait, ("done", result))
            except Exception as e:
                APP_LOGGER.exception("pipeline_error_in_background_thread")
                loop.call_soon_threadsafe(token_queue.put_nowait, ("error", str(e)))

        # Start pipeline in background thread
        executor_thread = threading.Thread(target=run_pipeline, daemon=True)
        executor_thread.start()

        # Send an immediate keep-alive comment to flush proxy headers on Render
        yield ": keepalive\n\n"

        # Consume tokens from queue and yield SSE events in real-time
        result = None
        while True:
            try:
                # Wait for up to 1.5 seconds without blocking the CPU
                msg_type, payload = await asyncio.wait_for(token_queue.get(), timeout=1.5)
            except asyncio.TimeoutError:
                if not executor_thread.is_alive():
                    if token_queue.empty():
                        break
                # Periodically yield ping comments to keep connection active and flush proxy buffer
                yield ": ping\n\n"
                continue

            if msg_type == "token":
                streamed_any = True
                sse_payload = json.dumps({"chunk": payload}, ensure_ascii=False)
                yield f"event: message\ndata: {sse_payload}\n\n"

            elif msg_type == "done":
                result = payload
                break

            elif msg_type == "error":
                err_payload = json.dumps({"error": str(payload)}, ensure_ascii=False)
                yield f"event: error\ndata: {err_payload}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                await asyncio.sleep(0.2)
                return

        if result is None:
            yield "event: done\ndata: [DONE]\n\n"
            return

        answer: str = result.get("answer") or ""
        metadata: dict = result.get("metadata", {})

        if not str(answer).strip():
            answer = (
                "Mình chưa thể tạo câu trả lời chi tiết từ dữ liệu hiện có. "
                "Bạn thử nêu rõ sở thích (thiên nhiên, văn hóa, ẩm thực) hoặc bán kính di chuyển để mình gợi ý tốt hơn."
            )

        APP_LOGGER.info(
            "sse_generation_ready answer_len=%s streamed_any=%s intent=%s detected_location=%s",
            len(answer),
            streamed_any,
            str(metadata.get("intent", "")),
            str(metadata.get("detected_location", "")),
        )

        # If no tokens were streamed (deterministic/unsupported provider), send full answer once
        if not streamed_any and answer:
            payload = json.dumps({"chunk": answer}, ensure_ascii=False)
            yield f"event: message\ndata: {payload}\n\n"
            await asyncio.sleep(0.1)  # Allow socket to flush response text

        # --- Build map locations ---
        source_nodes = (
            metadata.get("answered_route_nodes")
            or metadata.get("route_seed_nodes")
            or metadata.get("seed_nodes", [])
        )
        locations = []
        for node in source_nodes:
            lat = node.get("lat")
            lng = node.get("lng")
            if lat is not None and lng is not None:
                locations.append({
                    "id": node.get("id", ""),
                    "name": node.get("name", "Unknown"),
                    "labels": node.get("labels", []),
                    "type": (node.get("labels", [""])[0] if node.get("labels") else ""),
                    "coordinates": {"lat": float(lat), "lng": float(lng)},
                })

        locations = _filter_focus_locations(query, metadata, locations)
        if metadata.get("intent") != "TOUR_PLAN" and answer:
            locations = _filter_by_answer_mentions(answer, locations)

        meta_payload = json.dumps({
            "intent": str(metadata.get("intent", "")),
            "detected_location": metadata.get("detected_location", ""),
            "locations": locations,
            "graph": metadata.get("graph", {"nodes": [], "links": []}),
            "distance": metadata.get("distance"),
            "constraints": metadata.get("constraints", {}),
            "optimization_applied": bool(metadata.get("optimization_applied", False)),
            "tour_plan_safety": (
                _build_tour_plan_safety(metadata)
                if metadata.get("intent") == "TOUR_PLAN"
                else None
            ),
            # Constraint warning: sent when coastal/sunset/island requirement not fully met
            "constraint_warning": (
                metadata.get("constraint_warning")
                if metadata.get("intent") == "TOUR_PLAN"
                else None
            ),
            # Daily cluster plan: always sent for TOUR_PLAN so MapInterface can build day badges
            "daily_cluster_plan": (
                metadata.get("daily_cluster_plan", [])
                if metadata.get("intent") == "TOUR_PLAN"
                else []
            ),
            "metadata": (
                {
                    "nearby_mode": bool(metadata.get("nearby_mode", False)),
                    "max_hop_km": metadata.get("max_hop_km"),
                    "hop_distances_km": metadata.get("hop_distances_km", []),
                    "dropped_route_points": metadata.get("dropped_route_points", []),
                    "optimization_applied": bool(metadata.get("optimization_applied", False)),
                    "graph_ordering_applied": bool(metadata.get("graph_ordering_applied", False)),
                    "route_engine": metadata.get("route_engine"),
                    "daily_cluster_plan": metadata.get("daily_cluster_plan", []),
                    "route_optimizer_metrics": metadata.get("route_optimizer_metrics", {}),
                }
                if INCLUDE_DEBUG_METADATA
                else {}
            ),
        }, ensure_ascii=False)

        if metadata.get("intent") == "TOUR_PLAN":
            query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
            monitoring_payload = {
                "event": "route_optimizer_metrics",
                "env": APP_ENV,
                "query_hash": query_hash,
                "query_length": len(query or ""),
                "optimize_distance": bool(metadata.get("constraints", {}).get("optimize_distance", False)),
                "optimization_applied": bool(metadata.get("optimization_applied", False)),
                "nearby_mode": bool(metadata.get("nearby_mode", False)),
                "max_hop_km": metadata.get("max_hop_km"),
                "hop_distances_km": metadata.get("hop_distances_km", []),
                "dropped_route_points": metadata.get("dropped_route_points", []),
                "graph_ordering_applied": bool(metadata.get("graph_ordering_applied", False)),
                "route_engine": metadata.get("route_engine"),
                "daily_cluster_plan": metadata.get("daily_cluster_plan", []),
                "route_optimizer_metrics": metadata.get("route_optimizer_metrics", {}),
            }
            ROUTE_MONITOR_LOGGER.info(json.dumps(monitoring_payload, ensure_ascii=False))
        yield f"event: metadata\ndata: {meta_payload}\n\n"
        await asyncio.sleep(0.1)  # Allow socket to flush metadata

        yield "event: done\ndata: [DONE]\n\n"
        await asyncio.sleep(0.2)  # Critical delay on Linux to avoid TCP RST on connection close

    except Exception as e:
        APP_LOGGER.exception("sse_generator_error")
        err_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"event: error\ndata: {err_payload}\n\n"
        yield "event: done\ndata: [DONE]\n\n"
        await asyncio.sleep(0.2)


# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "GraphRAG API v2 — SSE Streaming đang chạy!"}


class PlaceSearchResponse(BaseModel):
    id: str
    name: str
    type: str
    address: str = ""
    location: Optional[List[float]] = None
    province: str = ""


@app.get("/api/places/search", response_model=List[PlaceSearchResponse])
def search_places(q: str = ""):
    q = (q or "").strip()
    if not q:
        return []

    from graph_rag.services.database import Neo4jService
    try:
        driver = Neo4jService.get_driver()
    except Exception as exc:
        APP_LOGGER.error("Failed to connect to Neo4j for search: %s", exc)
        return []

    query = """
    MATCH (n)
    WHERE (n:TouristAttraction OR n:Restaurant OR n:Accommodation)
      AND toLower(n.name) CONTAINS toLower($q)
    RETURN n.id AS id, n.name AS name, labels(n) AS labels, n.location AS location, n.address AS address, n.province AS province
    LIMIT 15
    """

    results = []
    try:
        with driver.session() as session:
            res = session.run(query, q=q)
            for record in res:
                labels = record["labels"] or []
                primary_label = labels[0] if labels else "Location"
                location = record["location"]
                
                coords = None
                if isinstance(location, list) and len(location) >= 2:
                    try:
                        coords = [float(location[0]), float(location[1])]
                    except (ValueError, TypeError):
                        pass
                
                results.append({
                    "id": str(record["id"] or ""),
                    "name": str(record["name"] or ""),
                    "type": str(primary_label),
                    "address": str(record["address"] or ""),
                    "location": coords,
                    "province": str(record["province"] or ""),
                })
    except Exception as exc:
        APP_LOGGER.error("Neo4j place search query error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database search failed: {exc}")

    return results


@app.post("/api/mapbox/directions")
async def mapbox_directions_proxy(request: Request, payload: DirectionsRequest):
    global _mapbox_daily_usage_date, _mapbox_daily_usage_count

    APP_LOGGER.info(
        "mapbox_directions_proxy received: points=%d, first=%s, profile=%s",
        len(payload.points),
        f"{payload.points[0].lng},{payload.points[0].lat}" if payload.points else "none",
        payload.profile,
    )

    mapbox_token = _resolve_mapbox_token()

    if not mapbox_token:
        raise HTTPException(
            status_code=500,
            detail="Missing MAPBOX_ACCESS_TOKEN on backend",
        )

    if len(payload.points) < 2:
        raise HTTPException(status_code=400, detail="At least 2 points are required")

    if len(payload.points) > 25:
        raise HTTPException(status_code=400, detail="Too many points (max 25)")

    now_ts = time.time()
    now_date = date.today().isoformat()
    client_ip = _normalize_client_ip(request)
    profile = _normalize_profile(payload.profile)
    cache_key = _build_route_cache_key(
        DirectionsRequest(
            points=payload.points,
            profile=profile,
            geometries=payload.geometries,
            overview=payload.overview,
        )
    )

    async with _mapbox_guard_lock:
        if _mapbox_daily_usage_date != now_date:
            _mapbox_daily_usage_date = now_date
            _mapbox_daily_usage_count = 0

        _prune_expired_route_cache(now_ts)

        # LRU eviction: remove oldest entries when cache exceeds max size
        while len(_mapbox_route_cache) > ROUTE_CACHE_MAX_SIZE:
            _mapbox_route_cache.popitem(last=False)

        cached_item = _mapbox_route_cache.get(cache_key)
        if cached_item:
            # LRU touch: move to end (most recently used)
            _mapbox_route_cache.move_to_end(cache_key)
            return {
                **cached_item["payload"],
                "cached": True,
                "usage": {
                    "daily_count": _mapbox_daily_usage_count,
                    "daily_limit": MAPBOX_INTERNAL_DAILY_LIMIT,
                },
            }

        user_window = _mapbox_rate_windows[client_ip]
        while user_window and now_ts - user_window[0] > 60:
            user_window.popleft()

        if len(user_window) >= MAPBOX_RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded (max requests per minute)",
            )

        if _mapbox_daily_usage_count >= MAPBOX_INTERNAL_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Quota exceeded (internal safe limit)",
            )

        user_window.append(now_ts)
        _mapbox_daily_usage_count += 1

    coord_string = ";".join([f"{p.lng},{p.lat}" for p in payload.points])
    qs = urllib.parse.urlencode(
        {
            "overview": payload.overview,
            "geometries": payload.geometries,
            "access_token": mapbox_token,
        }
    )
    route_url = f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{coord_string}?{qs}"
    APP_LOGGER.info("mapbox_directions_proxy calling: %s", route_url)

    try:
        with urllib.request.urlopen(route_url, timeout=12) as resp:
            raw_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        APP_LOGGER.error("Mapbox HTTP error %d: %s", exc.code, exc.reason)
        raise HTTPException(
            status_code=exc.code,
            detail=f"Mapbox HTTP error: {exc.reason}",
        ) from exc
    except urllib.error.URLError as exc:
        APP_LOGGER.error("Mapbox connection failed: %s", exc.reason)
        raise HTTPException(
            status_code=502,
            detail=f"Mapbox connection failed: {exc.reason}",
        ) from exc
    except Exception as exc:
        APP_LOGGER.error("Mapbox request unexpected error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Mapbox request failed: {exc}",
        ) from exc

    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Invalid response from Mapbox") from exc

    async with _mapbox_guard_lock:
        _mapbox_route_cache[cache_key] = {
            "expires_at": time.time() + MAPBOX_CACHE_TTL_SECONDS,
            "payload": data,
        }

    return {
        **data,
        "cached": False,
        "usage": {
            "daily_count": _mapbox_daily_usage_count,
            "daily_limit": MAPBOX_INTERNAL_DAILY_LIMIT,
        },
    }


_nominatim_cache: OrderedDict = OrderedDict()


def _reverse_geocode(coords_str: str) -> str:
    """Reverse geocode 'lat,lng' to a location name using Nominatim. Returns fallback on failure."""
    # Check cache first (LRU: move to end on hit)
    if coords_str in _nominatim_cache:
        _nominatim_cache.move_to_end(coords_str)
        APP_LOGGER.info("reverse_geocode cache hit for '%s'", coords_str)
        return _nominatim_cache[coords_str]

    try:
        parts = coords_str.split(",")
        lat, lng = float(parts[0].strip()), float(parts[1].strip())
        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lng}&format=json&accept-language=vi&zoom=10"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "graphrag-travel-bot/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            addr = data.get("address", {})
            # Prefer city/town/village, fallback to state
            name = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("state") or ""
            if name:
                _nominatim_cache[coords_str] = name
                # LRU eviction: remove oldest entry if over limit
                if len(_nominatim_cache) > NOMINATIM_CACHE_MAX_SIZE:
                    _nominatim_cache.popitem(last=False)
                APP_LOGGER.info("reverse_geocode '%s' -> '%s'", coords_str, name)
                return name
    except Exception as exc:
        APP_LOGGER.warning("reverse_geocode failed for '%s': %s", coords_str, exc)
        # Try cache as fallback on failure
        if coords_str in _nominatim_cache:
            APP_LOGGER.info("reverse_geocode fallback cache hit for '%s'", coords_str)
            return _nominatim_cache[coords_str]
    return "Vị trí không xác định"


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/chat")
@limiter.limit("5/minute")
async def chat_endpoint(request: Request, payload: ChatRequest):
    from graph_rag.utils.text import clean_query_format
    query = clean_query_format(payload.query)
    if not query.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    history = [{"role": m.role, "content": m.content} for m in payload.chat_history]
    
    # Update payload query with cleaned version for sse_generator
    payload.query = query

    # Reverse geocode GPS "lat,lng" to location name if needed.
    current_location = payload.current_location.strip()
    user_gps = ""
    if current_location and "," in current_location:
        user_gps = current_location  # preserve raw GPS for distance calc
        current_location = _reverse_geocode(current_location)

    return StreamingResponse(
        sse_generator(payload.query, history, current_location=current_location, user_gps=user_gps),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

