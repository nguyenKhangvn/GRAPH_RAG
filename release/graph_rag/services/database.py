from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
import logging

logger = logging.getLogger(__name__)

import threading
import os
from neo4j import GraphDatabase
from graph_rag.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jService:
    _driver = None
    # Tạo khóa (Lock) để đảm bảo Thread-Safety
    _lock = threading.Lock()

    @classmethod
    def _build_connection_candidates(cls):
        """Build ordered Neo4j connection candidates (primary -> fallbacks)."""
        candidates = []

        primary = (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        candidates.append(primary)

        # Optional explicit fallback from env.
        fb_uri = os.getenv("NEO4J_FALLBACK_URI")
        fb_pwd = os.getenv("NEO4J_FALLBACK_PASSWORD")
        if fb_uri and fb_pwd:
            candidates.append(
                (
                    fb_uri,
                    os.getenv("NEO4J_FALLBACK_USER", "neo4j"),
                    fb_pwd,
                )
            )

        # Safe local fallback when primary is cloud/remote.
        # Use 127.0.0.1 to avoid Windows resolving localhost to ::1 first.
        local_fallback_pwd = os.getenv("NEO4J_LOCAL_FALLBACK_PASSWORD")
        if "localhost" not in (NEO4J_URI or "") and "127.0.0.1" not in (NEO4J_URI or ""):
            candidates.append(("bolt://127.0.0.1:7687", NEO4J_USER, NEO4J_PASSWORD))
            if local_fallback_pwd:
                candidates.append(("bolt://127.0.0.1:7687", "neo4j", local_fallback_pwd))

        # Deduplicate while preserving order.
        deduped = []
        seen = set()
        for uri, user, pwd in candidates:
            key = (uri, user, pwd)
            if uri and key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    @classmethod
    def get_driver(cls):
        """
        Singleton Pattern với Double-Checked Locking.
        Đảm bảo an toàn khi chạy nhiều luồng (Multi-threading).
        """
        # 1. Check lần đầu (Nhanh): Nếu đã có driver rồi thì trả về luôn, không cần chờ Lock
        if cls._driver is None:
            
            # 2. Acquire Lock: Chỉ cho phép 1 thread đi vào đoạn code này tại một thời điểm
            with cls._lock:
                
                # 3. Check lần hai (Double-check): 
                # Đề phòng trường hợp trong lúc Thread A chờ Lock, Thread B đã kịp tạo driver rồi.
                if cls._driver is None:
                    logger.info("Connecting to Neo4j...")
                    last_error = None
                    for uri, user, password in cls._build_connection_candidates():
                        try:
                            driver = GraphDatabase.driver(
                                uri,
                                auth=(user, password),
                                notifications_disabled_categories=["DEPRECATION", "UNRECOGNIZED"],
                            )
                            driver.verify_connectivity()
                            logger.info(" Connected to Neo4j successfully via %s", uri)
                            cls._driver = driver
                            break
                        except (Neo4jClientError, ServiceUnavailable) as e:
                            last_error = e
                            logger.error(" Failed to connect via %s: %s", uri, e)

                    if cls._driver is None:
                        raise last_error
                        
        return cls._driver

    @classmethod
    def close_driver(cls):
        """
        Đóng kết nối an toàn.
        """
        with cls._lock: # Lock luôn khi đóng để tránh tranh chấp
            if cls._driver:
                cls._driver.close()
                cls._driver = None
                logger.info("Neo4j Connection Closed.")
