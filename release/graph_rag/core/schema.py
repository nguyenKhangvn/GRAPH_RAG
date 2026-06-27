# graph_rag/core/schema.py

from enum import Enum

class EntityType(str, Enum):
    RESTAURANT          = "Restaurant"
    TOURIST_ATTRACTION  = "TouristAttraction"
    ACCOMMODATION       = "Accommodation"
    EVENT               = "Event"
    TOUR                = "Tour"
    DISH                = "Dish"
    TRAVEL_AGENCY       = "TravelAgency"
    TRAVEL_INFO         = "TravelInfo"  # Thông tin giao thông, sân bay, etc.
    LOCATION            = "Location"   # Xã / Phường – trích xuất từ địa chỉ
    CATEGORY            = "Category"   # Phân loại TouristAttraction
    SPECIALTY           = "Specialty"  # Đặc sản vùng miền

class GraphSchema:
    """
    Source of Truth cho Schema của hệ thống GraphRAG.
    Chứa định nghĩa về Entities, Relationships và Logic Prompting.

    Thống kê thực thể (2026-06-27, query trực tiếp Neo4j):
        Accommodation    : 360
        TouristAttraction: 201
        Dish             : 151
        Restaurant       : 132
        Location         :  89
        TravelAgency     :  73
        Specialty        :  49
        Tour             :  36
        TravelInfo       :  32
        Event            :  18
        Category         :  10
        ─────────────────────
        TỔNG             : 1 152

    Thống kê quan hệ (12 loại — RELATIONSHIPS list):
        NEAR          : 3 708  (Restaurant/Accommodation → TouristAttraction)
        LOCATED_IN    : 1 773  (Entity → Location)
        INCLUDES      :   206  (Tour → TouristAttraction)
        BELONGS_TO    :   192  (TouristAttraction → Category)
        HAS           :   169  (Restaurant → Dish 132 + Restaurant → Specialty 37)
        OFFERS        :    36  (TravelAgency → Tour)
        HELD_AT       :    21  (Event → TouristAttraction)
        Guide_for     :    32  (TravelInfo → Location)
        SUPERSEDED_BY :     1  (Location → Location)
        ─────────────────────
        Đã xóa       : SPECIALTY_OF, Event/TravelInfo LOCATED_IN (chuyển TravelInfo sang Guide_for)

    location property types:
        - WGS84Point: Restaurant, Accommodation, TouristAttraction (geo coordinates)
        - String    : Dish, Specialty, TravelInfo (tên tỉnh: "Tỉnh Gia Lai", "Bình Định")
        - Không có  : Event, Location, Category

    TravelInfo location: property → Guide_for relationship
        - TravelInfo có t.location/t.province properties + Guide_for → Location
        - Event: location được suy ra qua HELD_AT → TouristAttraction → LOCATED_IN → Location

    Data gaps (cần bổ sung):
        - Dish: 102/151 thiếu category, location, region_group, province, embedding
        - Restaurant: phone (27%), email (25%) thiếu nhiều
        - Accommodation: amenities/capacity/villa_segment có thể rỗng

    Data enrichment (2026-06-13):
        - enriched_rating: 132/132 Restaurant, 360/360 Accommodation, 198/201 TouristAttraction
        - opening_hours: 98/132 Restaurant, 192/201 TouristAttraction
        - Nguồn: SerpAPI, estimated, WebSearch/manual
    """

    # 1. Các thực thể (Dùng Enum values)
    ENTITIES = [e.value for e in EntityType]

    # 2. Logic Quan hệ — khớp DB thực tế (2026-06-27)
    RELATIONSHIPS = [
        # --- Địa lý (LOCATED_IN) ---
        # Event: location suy ra qua HELD_AT → TouristAttraction → LOCATED_IN → Location
        "(:Accommodation)-[:LOCATED_IN]->(:Location)",        # 868 rels
        "(:Restaurant)-[:LOCATED_IN]->(:Location)",           # 316 rels
        "(:TouristAttraction)-[:LOCATED_IN]->(:Location)",    # 511 rels
        "(:Location)-[:SUPERSEDED_BY]->(:Location)",          # 1 rel (post-2025 merger)

        # --- Phân loại ---
        "(:TouristAttraction)-[:BELONGS_TO]->(:Category)",    # 192 rels

        # --- Lân cận (NEAR) ---
        "(:Restaurant)-[:NEAR]->(:TouristAttraction)",        # 849 rels
        "(:Accommodation)-[:NEAR]->(:TouristAttraction)",     # 2856 rels

        # --- Ẩm thực ---
        "(:Restaurant)-[:HAS]->(:Dish)",                      # 132 rels
        "(:Restaurant)-[:HAS]->(:Specialty)",                 # 37 rels (Specialty also labeled Dish)

        # --- Sự kiện ---
        "(:Event)-[:HELD_AT]->(:TouristAttraction)",          # 21 rels
        # Event location chain: (Event)-[:HELD_AT]->(TouristAttraction)-[:LOCATED_IN]->(Location)

        # --- Thông tin hướng dẫn ---
        "(:TravelInfo)-[:Guide_for]->(:Location)",            # 32 rels

        # --- Tour & Đại lý ---
        "(:Tour)-[:INCLUDES]->(:TouristAttraction)",          # 206 rels
        "(:TravelAgency)-[:OFFERS]->(:Tour)",                 # 36 rels
    ]

    # 3. Property Whitelist — khớp DB thực tế (2026-06-13)
    # Lưu ý: KHÔNG thêm property không tồn tại (Accommodation.email = 0%)
    PROPERTIES = {
        EntityType.RESTAURANT:         ["id", "name", "location", "address", "phone", "type", "tags", "opening_hours", "province", "email", "enriched_rating", "enriched_rating_source"],
        EntityType.TOURIST_ATTRACTION: ["id", "name", "description", "location", "address", "category", "ticket_price", "opening_hours", "phone", "province", "enriched_rating", "enriched_rating_source"],
        EntityType.ACCOMMODATION:      ["id", "name", "description", "location", "address", "phone", "type", "price_range", "amenities", "capacity", "villa_segment", "province", "enriched_rating", "enriched_rating_source"],
        EntityType.EVENT:              ["id", "name", "address", "category", "month", "activities"],
        EntityType.TRAVEL_AGENCY:      ["id", "name", "address", "phone", "email", "website"],
        EntityType.TOUR:               ["id", "name", "description", "price", "duration", "start_location"],
        EntityType.DISH:               ["id", "name", "description", "category", "location", "region_group", "province"],
        EntityType.SPECIALTY:          ["id", "name", "description", "category", "location", "region_group", "province"],
        EntityType.TRAVEL_INFO:        ["id", "name", "description", "topic", "location", "province"],
        EntityType.LOCATION:           ["id", "name", "region_group", "admin_level", "admin_status", "legacy_district", "current_province", "old_units", "aliases"],
        EntityType.CATEGORY:           ["name"],
    }

    # 4. ENUMS (Danh sách giá trị cố định – dùng để lọc chính xác trong Cypher)
    # Category: node Category trong Neo4j (phân loại TouristAttraction)
    # AccommodationType: trường .type trên node Accommodation (không phải node riêng)
    # RestaurantType: trường .type trên node Restaurant
    # EventCategory: trường .category trên node Event
    # TravelInfoTopic: trường .topic trên node TravelInfo
    CATEGORICAL_VALUES = {
        "Category": [
            "Danh lam thắng cảnh", "Di tích lịch sử - Văn hóa", "Di tích lịch sử - văn hóa",
            "Làng nghề - Nông nghiệp", "Làng nghề - Văn hóa", "Làng nghề truyền thống",
            "Làng văn hóa", "Thắng cảnh thiên nhiên", "Điểm check-in", "Điểm tham quan"
        ],
        "AccommodationType": [
            "Nhà nghỉ du lịch", "Khách sạn", "Nhà nghỉ", "Khách Sạn",
            "Homestay", "Resort", "Nhà Khách", "Villa"
        ],
        "RestaurantType": [
            "Nhà hàng", "Quán ăn/Đặc sản", "Cafe/Đồ uống"
        ],
        "EventCategory": [
            "Lễ hội văn hóa dân gian", "Lễ hội văn hóa tín ngưỡng", "Lễ hội văn hóa",
            "Lễ hội lịch sử", "Giải chạy địa hình", "Lễ hội Du lịch",
            "Lễ hội Văn hóa - Du lịch", "Lễ hội văn hóa - du lịch", "Giải chạy marathon"
        ],
        "TravelInfoTopic": [
            "transport", "emergency", "weather", "budget", "accommodation_tips",
            "airport", "shopping", "payment", "health", "community", "event"
        ],
    }

    @classmethod
    def get_system_prompt_context(cls) -> str:
        """
        Tạo context string tối ưu hóa token để nhúng vào System Prompt.
        """
        return f"""
                ### KNOWLEDGE GRAPH SCHEMA

                1. **Entities** ({len(cls.ENTITIES)}):
                {", ".join(cls.ENTITIES)}

                2. **Relationship Paths** (Reasoning Logic):
                {chr(10).join([f"   - {r}" for r in cls.RELATIONSHIPS])}

                3. **Categorical Filters** (Exact Match Required):
                - Attraction Categories (Category node): {cls.CATEGORICAL_VALUES['Category']}
                - Accommodation Types (.type property): {cls.CATEGORICAL_VALUES['AccommodationType']}
                - Restaurant Types (.type property): {cls.CATEGORICAL_VALUES['RestaurantType']}
                - Event Categories (.category property): {cls.CATEGORICAL_VALUES['EventCategory']}
                - TravelInfo Topics (.topic property): {cls.CATEGORICAL_VALUES['TravelInfoTopic']}

                4. **Traversal Strategy**:
                - Vị trí       : (Entity)-[:LOCATED_IN]->(Location) — Accommodation, Restaurant, TouristAttraction
                - Gần đó       : (Restaurant|Accommodation)-[:NEAR]->(TouristAttraction)
                - Loại hình    : (TouristAttraction)-[:BELONGS_TO]->(Category)
                - Món ăn       : (Restaurant)-[:HAS]->(Dish) hoặc (Restaurant)-[:HAS]->(Specialty). Dish/Specialty: s.location, s.province
                - Đặc sản      : (Restaurant)-[:HAS]->(Specialty). Dùng s.location/s.province để filter theo vùng
                - Tour         : (TravelAgency)-[:OFFERS]->(Tour)-[:INCLUDES]->(TouristAttraction)
                - Sự kiện      : (Event)-[:HELD_AT]->(TouristAttraction). Location suy ra qua TouristAttraction→LOCATED_IN
                - TravelInfo   : (TravelInfo)-[:Guide_for]->(Location). Traverse: (TravelInfo)-[:Guide_for]->(Location)<-[:LOCATED_IN]-(Entity)
                - Admin merger  : (Location)-[:SUPERSEDED_BY]->(Location)
                """
