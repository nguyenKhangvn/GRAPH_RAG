# Data Catalog — Dữ liệu Du lịch Gia Lai / Bình Định

> **Mô tả:** Tổng hợp toàn bộ dữ liệu thô thu thập được từ các nguồn khác nhau
> phục vụ đồ án GraphRAG cho du lịch tỉnh Gia Lai và Bình Định.
>
> **Domain:** Ẩm thực, tham quan, lưu trú, sự kiện, tour, giao thông
> **Phạm vi địa lý:** Gia Lai, Bình Định
> **Số lượng entities trong Neo4j (hiện tại):** 1,102 nodes

---

## Tổng quan các nguồn dữ liệu

| # | Nguồn | Website | Loại dữ liệu | Phương pháp | Số lượng |
|---|-------|---------|-------------|-------------|---------|
| 1 | **Sở VHTTDL Bình Định** | [quanlyluhanh.vn](https://quanlyluhanh.vn) | TravelAgency | Crawl cổng quản lý lữ hành | 74 |
| 2 | **Google Places API** | [maps.googleapis.com](https://maps.googleapis.com) | Restaurant, Accommodation, TouristAttraction | API crawl (rating, hours) | 132+361+198 |
| 3 | **SerpAPI (Google Maps)** | [serpapi.com](https://serpapi.com) | Restaurant, Accommodation, TouristAttraction | API crawl (cross-validate) | 132+361+198 |
| 4 | **Tourism web (3 sites)** | [travel.com.vn](https://travel.com.vn), [vntrip.vn](https://vntrip.vn), [vietnamtourism.vn](https://vietnamtourism.vn) | TouristAttraction | Crawl rating + opening hours | 198 |
| 5 | **Cổng tỉnh Gia Lai** | [gialai.gov.vn](https://gialai.gov.vn/du-khach/) | Dish / Specialty, Event, TouristAttraction | Crawl từ chuyên mục "Du khách" | ~100 |
| 6 | **Cổng du lịch Quy Nhơn** | [dulichquynhon.binhdinh.gov.vn](http://dulichquynhon.binhdinh.gov.vn) | TouristAttraction | Crawl từng địa danh Bình Định | ~80 |
| 7 | **Quy Nhơn Tourist** | [quynhontourist.com](https://quynhontourist.com) | Tour | Crawl tour + lịch trình | 36 |
| 8 | **Wikipedia / VnExpress** | [vi.wikipedia.org](https://vi.wikipedia.org), [vnexpress.net](https://vnexpress.net) | TouristAttraction | Bổ sung thông tin vài điểm 
| 9 | **Travel info tổng hợp** | Nhiều nguồn | TravelInfo | Sân bay, thời tiết, taxi, cấp cứu… | 32 |
| 10 | **Địa chính** | Thống kê | Location | Xử lý phường/xã Gia Lai | 89 |

---

## 1. Dữ liệu TravelAgency (74 công ty lữ hành)

### Nguồn: Sở Văn hóa Thể thao và Du lịch Bình Định

- **File gốc:** `dont_use/data_for_neo4j/agency/quanlyluhanh_agencies.json`
- **Phương pháp:** Crawl từ [quanlyluhanh.vn](https://quanlyluhanh.vn) — cổng quản lý lữ hành của Sở VHTTDL
- **Nội dung:** Danh sách công ty lữ hành được cấp phép tại Bình Định
- **Fields:** `nameVietnamese, nameEnglish, type, address, licenseNumber, issueDate, phone, fax, website, email`
- **Import script:** `data_neo4j_v1/travelAgency/import_4_agencies_simple.py`
- **Entity trong Neo4j:** TravelAgency — 73 nodes (1 bị lọc do thiếu thông tin)

### Các phiên bản:
| Version | File | Số lượng | Ghi chú |
|---------|------|---------|---------|
| V1 | `data_neo4j_v1/travelAgency/agency.json` | 74 | Raw từ crawl |
| Final | Neo4j | 73 | Sau import |

---

## 2. Dữ liệu Restaurant (132 nhà hàng / quán ăn)

### Nguồn 1: Google Places API (raw)
- **File:** `data_v2/raw/google_places/restaurant.json` — 132 entries
- **Phương pháp:** Gọi Google Places API với tên quán từ danh sách gốc
- **Fields:** `id, name, neo4j_address, google_rating, google_ratings_total, google_opening_hours, google_formatted_address, source, crawled_at`
- **Script crawl:** `data_v2/scripts/crawl_google_places.py`

### Nguồn 2: SerpAPI (Google Maps engine) — cross-validation
- **File:** `data_v2/raw/serpapi/restaurant.json` — 132 entries
- **Phương pháp:** SerpAPI Google Maps engine để lấy rating + review count
- **Fields:** `id, name, neo4j_address, serpapi_rating, serpapi_rating_count, serpapi_opening_hours, serpapi_price, serpapi_formatted_address, serpapi_lat, serpapi_lng`
- **Script crawl:** `data_v2/scripts/crawl_serpapi.py`

### Enriched (processed — đã cross-validate)
- **File:** `data_v2/processed/restaurant_enrichment.json` — 132 entries
- **Kết quả:**
  - 132 restaurant được enriched rating (từ SerpAPI)
  - 98 enriched opening_hours (từ Google Places)
  - Cross-validation: 0 issues
- **Fields:** Kết hợp của Google + SerpAPI + `enriched_rating, enriched_rating_sources`
- **Script xử lý:** `data_v2/scripts/validate_data.py`

### Import vào Neo4j:
- **Script:** `data_v2/scripts/import_to_neo4j.py`
- **Entity:** Restaurant — 132 nodes
- **Properties thêm:** `enriched_rating, enriched_rating_source, google_rating, serpapi_rating, opening_hours, cross_validation_flag`

### Các phiên bản dữ liệu:
| Version | File | Số lượng | Ghi chú |
|---------|------|---------|---------|
| Gốc | `dont_use/data_for_neo4j/restaurant/restaurant_normalized.json` | 121 | Crawl từ web |
| V1 | `data_neo4j_v1/restaurant/restaurant_id_address.json` | 121 + thêm | 121 raw + enriched |
| V2 | `data_neo4j_v2/restaurant/restaurant_v2.json` | 133 | Thêm 11 quán, fix category |
| V3 | `data_neo4j_v3/restaurant/new_restaurants.json` | 8 | Bổ sung thủ công |
| Final | Neo4j | 132 | Sau merge + cleanup |

---

## 3. Dữ liệu Accommodation (360 khách sạn, homestay)

### Nguồn 1: Google Places API (raw)
- **File:** `data_v2/raw/google_places/accommodation.json` — 361 entries
- **Phương pháp:** Google Places API
- **Fields:** `id, name, neo4j_address, google_rating, google_ratings_total, google_opening_hours, google_price_level`
- **Script:** `data_v2/scripts/crawl_google_places.py`

### Nguồn 2: SerpAPI (cross-validation) — từ Google Maps engine
- **File:** `data_v2/raw/serpapi/accommodation.json` — 361 entries
- **Fields:** `id, name, neo4j_address, serpapi_rating, serpapi_rating_count, serpapi_opening_hours, serpapi_price, serpapi_formatted_address, serpapi_lat, serpapi_lng`
- **Script:** `data_v2/scripts/crawl_serpapi.py`

### Enriched (processed):
- **File:** `data_v2/processed/accommodation_enrichment.json` — 361 entries
- **Kết quả:**
  - 254 accommodations có enriched_rating (từ SerpAPI)
  - 7 enriched opening_hours
  - 46 enriched phone, 9 enriched email
- **Script:** `data_v2/scripts/validate_data.py`

### Các phiên bản:
| Version | File | Số lượng | Ghi chú |
|---------|------|---------|---------|
| Gốc | `dont_use/data_for_neo4j/accommodation/data_v4.json` | 320 | Crawl từ web |
| V1 | `data_neo4j_v1/accommodation/data_v1_fixed.json` | ~361 | Thêm + fix |
| V2 | `data_neo4j_v2/accommodation/accommodation_v2.json` | 362 | Thêm price_range, star_rating |
| V3 | `data_neo4j_v3/accommodation/new_accommodation.json` | 10 | Bổ sung thủ công |
| Final | Neo4j | 360 | Sau merge + cleanup |

---

## 4. Dữ liệu TouristAttraction (201 điểm tham quan)

### Nguồn gốc: Web crawl tổng hợp
- **File gốc:** `dont_use/data_for_neo4j/tourist_atractions/merged_tourist_attractions.json` — 153 entries
- **Phương pháp:** Crawl từ các website du lịch, tourism boards
- **Fields gốc:** `id, name, description, address, district_hint, category, ticketPrice, openingHours, latitude, longitude, sourceUrl, image, location, url, fullAddress`
- **Các website cụ thể trong data:**
  - **[dulichquynhon.binhdinh.gov.vn](http://dulichquynhon.binhdinh.gov.vn)** — Cổng thông tin du lịch Quy Nhơn, Bình Định (~80 địa danh: bãi biển, tháp Chăm, di tích lịch sử, đền chùa, làng nghề…)
    - VD: `/vi/bienkyco`, `/vi/ghenhrang`, `/vi/thapdoiquynhon`, `/vi/langchaihaiminh`, `/vi/baibienlodieu`…
    - Crawl từng URL cụ thể theo ID, parse tên, địa chỉ, mô tả, hình ảnh
  - **[gialai.gov.vn](https://gialai.gov.vn/du-khach/)** — Chuyên mục "Du khách" của cổng tỉnh Gia Lai (~60 địa danh, sự kiện, đặc sản)
    - VD: `/du-khach/danh-lam-thang-canh/`, `/du-khach/di-tich-lich-su-van-hoa/`, `/du-khach/le-hoi/`
    - Dùng chung domain gialai.gov.vn cho cả Dish/Specialty, Event, TouristAttraction
  - **[vi.wikipedia.org](https://vi.wikipedia.org)** — 1 địa danh (Thác Phú Cường)
  - **[vnexpress.net](https://vnexpress.net)** — 1 địa danh

### Enrichment V1: thêm ticket_price, opening_hours, phone
- **File:** `data_neo4j_v1/tourist/final_tourist_data_enriched.json` — 192 entries
- **Phương pháp:** Tổng hợp từ Google Places + SerpAPI + crawl bổ sung
- **Fields thêm:** `ticket_price, opening_hours, phone, phone_verified`

### Enrichment V2: thêm region_focus, province mapping
- **File:** `data_neo4j_v2/tourist/tourist_enriched_v2.json` — 192 entries
- **Fields thêm:** `ticket_price, opening_hours, phone, province, region, legacy_province, region_focus`

### Manual enrichment (thủ công)
- **Template:** `data_v2/manual/touristattraction_template.json` — 198 entries
- **Phương pháp:** Tạo template → fill thủ công rating, opening_hours từ web
- **File review:** `data_v2/review/attraction_review.json`
- **Script:** `data_v2/scripts/create_manual_template.py`, `data_v2/scripts/merge_manual_data.py`

### Google Places API (enrich rating)
- **File:** `data_v2/raw/google_places/touristattraction.json` — 198 entries
- **Script:** `data_v2/scripts/crawl_google_places.py`

### SerpAPI (enrich rating)
- **File:** `data_v2/raw/serpapi/touristattraction.json` — 198 entries
- **Script:** `data_v2/scripts/crawl_serpapi.py`
- **Lưu ý:** SerpAPI rating cho tourist attraction = 0 (không tìm thấy)

### V3: bổ sung
- **File:** `data_neo4j_v3/tourist/new_tourist.json` — 3 entries
- **Script:** `data_neo4j_v3/import_v3.py`

**Final trong Neo4j:** TouristAttraction — **201 nodes**

---

## 5. Dữ liệu Dish / Specialty (151 món ăn / 49 đặc sản)

### Nguồn gốc: Web crawl ẩm thực
- **File gốc:** `dont_use/data_for_neo4j/specialty/dishes.json` — 39 entries
- **Phương pháp:** Crawl từ website review ẩm thực, blog du lịch
- **Fields gốc:** `name, description, brief, url, full_url, image, location, category`
- **Website chính:** **[gialai.gov.vn](https://gialai.gov.vn/du-khach/dac-san-gia-lai/)** — chuyên mục "Đặc sản Gia Lai" + "Đặc sản Bình Định"
  - VD: `/du-khach/dac-san-gia-lai/pho-kho-gia-lai.html`, `/du-khach/dac-san-binh-dinh/ca-o-cuon-banh-trang3.html`
  - Mỗi món có url riêng trên gialai.gov.vn với mô tả, hình ảnh

### V1: Import dish + restaurant mapping
- **File:** `data_neo4j_v1/dish/dishes.json` — 39 món
- **File mapping:** `data_neo4j_v1/dish/restaurant_dish_mapping_FINAL.json` — mapping quán → món
- **File:** `data_neo4j_v1/dish/dish_v1.json` — 39 món enriched
- **Script:** `data_neo4j_v1/dish/import_final_full_process.py`

### V2: Bổ sung món thiếu
- **File:** `data_neo4j_v2/dish/dishes_v2.json` — 50 entries
- **Thêm:** Cà phê Pleiku, Gỏi lá rừng, Bún chả cá Quy Nhơn, Bánh xèo tôm nhảy, Nem Chợ Huyện + phân biệt Gia Lai/Bình Định
- **Fields thêm:** `province, region, legacy_province, region_focus`
- **Script:** `data_neo4j_v2/dish/import_dish_v2.py`
- **Script embedding:** `data_neo4j_v2/dish/generate_dish_embeddings.py`

### Specialty (dual label)
- **File:** `data_neo4j_v2/dish/dishes_v2.json` chứa cả Specialty và Dish
- **49 nodes** có label `Specialty` + `Dish`, 102 nodes còn lại chỉ `Dish`
- Properties: `id, name, description, category, location, region_group, province`

**Final trong Neo4j:** Dish — **151 nodes** (49 Specialty + 102 non-Specialty)

---

## 6. Dữ liệu Event (18 sự kiện / lễ hội)

### Nguồn gốc: Web crawl lễ hội
- **File gốc:** `dont_use/data_for_neo4j/event/festivals_processed.json`
- **Phương pháp:** Crawl từ website du lịch, lịch festival Bình Định / Gia Lai
- **Fields:** `id, name, address, category, month, activities, year, province`
- **Website:** **[gialai.gov.vn](https://gialai.gov.vn/du-khach/le-hoi/)** — chuyên mục "Lễ hội"
  - VD: `/du-khach/le-hoi/le-hoi-dam-trau-mung-nha-rong-moi3.html`, `/du-khach/le-hoi/festival-van-hoa-cong-chieng-tay-nguyen.html`
  - Các lễ hội đặc trưng Tây Nguyên: cồng chiêng, cầu mưa, đua thuyền độc mộc…

### V1 → V2:
| Version | File | Số lượng | Ghi chú |
|---------|------|---------|---------|
| Gốc | `dont_use/data_for_neo4j/event/festivals_processed.json` | ~24 | Raw crawl |
| V1 | `data_neo4j_v1/event/event.json` | 24 events | |
| V2 | `data_neo4j_v2/event/event_v2.json` | 19 | Thêm year, fix location |

**Final trong Neo4j:** Event — **18 nodes** (sau cleanup)

---

## 7. Dữ liệu Tour (36 tour du lịch)

### Nguồn gốc: Web crawl từ tour operator
- **File gốc:** `dont_use/data_for_neo4j/tour/quynhontourist/tours_with_schedule.json` — tour + lịch trình
- **Phương pháp:** Crawl từ website [quynhontourist.com](https://quynhontourist.com) và các tour operator Bình Định
- **Fields:** `id, schedule_id, title, duration, price, start_location, included, excluded, schedule`

### V1 → V2:
| Version | File | Số lượng | Ghi chú |
|---------|------|---------|---------|
| Gốc | `dont_use/data_for_neo4j/tour/quynhontourist/` | ~40 | Raw |
| V1 | `data_neo4j_v1/tour/tours_with_schedule.json` | 36 tours | Có lịch trình |
| V1 | `data_neo4j_v1/tour/tours_relationship.json` | 36 tours | Chỉ ID + schedule |
| V2 | `data_neo4j_v2/tour/tours_v2.json` | ~36 | Province mapping |

**Final trong Neo4j:** Tour — **36 nodes**
- Relationship: `(Tour)-[:INCLUDES]->(TouristAttraction)` — 206 rels
- `(TravelAgency)-[:OFFERS]->(Tour)` — 36 rels

---

## 8. Dữ liệu TravelInfo (32 thông tin du lịch)

### Nguồn gốc: Web crawl tổng hợp
- **File:** `data_neo4j_v2/travel_info/travel_info.json` — 29 entries
- **Phương pháp:** Thu thập từ các nguồn: sân bay, bến xe, thời tiết, y tế, taxi, số điện thoại khẩn cấp
- **Topics:** Di chuyển (sân bay Pleiku, Quy Nhơn), thời tiết, phương tiện, thanh toán, cấp cứu, tiêm phòng

### V3 bổ sung:
- **File:** `data_neo4j_v3/travel_info/new_travel_info.json` — 3 entries
- **Fields:** `id, topic, name, description, location, contact`

**Final trong Neo4j:** TravelInfo — **32 nodes**
- Relationship: `(TravelInfo)-[:Guide_for]->(Location)` — 32 rels

---

## 9. Dữ liệu Location (89 đơn vị hành chính)

### Nguồn: Thống kê phường/xã Gia Lai
- **File:** `data_neo4j_v1/tourist/thongke_phuong_xa.json` — raw district/ward data
- **Phương pháp:** Xử lý từ dữ liệu địa chính, phân cấp hành chính
- **Script xử lý:** `data_neo4j_v1/tourist/tk.py`, `data_neo4j_v1/tourist/import_1_attractions.py`

### Scripts xử lý địa lý:
| Script | Mục đích |
|--------|---------|
| `data_neo4j_v2/scripts/import_province.py` | Import province hierarchy |
| `data_neo4j_v2/scripts/merge_duplicate_locations.py` | Merge Location trùng |
| `data_neo4j_v2/scripts/fix_admin_levels.py` | Fix admin_level (province/ward/area) |
| `data_neo4j_v2/scripts/migrate_region_focus.py` | Mapping region_focus |

### Cấu trúc Location:
| admin_level | Số lượng | Ví dụ |
|------------|---------|-------|
| province | 3 | Gia Lai, Bình Định, Kon Tum |
| area | 3 | Pleiku, An Nhơn, Quy Nhơn |
| ward | 83 | Các phường/xã |

**Final trong Neo4j:** Location — **89 nodes** (50 current + 39 merged)
- `(Location)-[:SUPERSEDED_BY]->(Location)` — 1 rel (post-2025 merger)

---

## 10. Dữ liệu bổ sung từ Data V2 Enrichment

### Review / Rating data (SerpAPI review crawl)
- **File:** `dont_use/data_for_neo4j/review/cy.json` — review crawl tổng hợp
- **File:** `dont_use/data_for_neo4j/review/ddnhahang_final_id.json` — review nhà hàng
- **File:** `dont_use/data_for_neo4j/review/nhahang_final_id.json` — review nhà hàng
- **File:** `dont_use/data_for_neo4j/review/thamquan_serpapi_final.json` — review điểm tham quan

### Kết quả validated (audit trail):
| File | Nội dung |
|------|---------|
| `data_v2/audit/validation_report.json` | Cross-validation report |
| `data_v2/audit/import_log.json` | Import log |
| `data_v2/review/accommodation_review.json` | Review data accommodation |
| `data_v2/review/restaurant_review.json` | Review data restaurant |
| `data_v2/review/attraction_review.json` | Review data attraction |

---

## Tổng hợp phiên bản dữ liệu

### V1 — Initial crawl (12/2025)
- **Dữ liệu:** Crawl lần đầu từ các web du lịch
- **Entities:** TouristAttraction (153), Restaurant (121), Accommodation (320), Event (~24), Dish (39), TravelAgency (74), Location (89)
- **Scripts:** `data_neo4j_v1/`
- **Lưu ý:** Thiếu nhiều field (rating, opening_hours, ticket_price)

### V2 — Schema mở rộng (06/2026)
- **Bổ sung:** TravelInfo (29), thêm Dish (50), Event (19), province/region mapping
- **Enrichment:** Rating từ Google Places + SerpAPI, cross-validation
- **Scripts:** `data_neo4j_v2/`, `data_v2/`
- **Thay đổi:** Thêm price_range, star_rating, ticket_price, opening_hours, province, region_focus

### V3 — Fix thủ công (06/2026)
- **Bổ sung:** Missing entities (3-10 mỗi loại)
- **Script:** `data_neo4j_v3/import_v3.py`
- **Fix:** Province mapping, location hierarchy

### GraphRAG (hiện tại)
- **Entities:** 1,152 nodes (11 loại)
- **Relationships:** 12 loại, ~6,140 rels
- **Enriched properties:** rating, opening_hours, ticket_price cho nhiều entity

---

## Đường dẫn file

```
data_v2/                         ← Crawl + Enrichment data
├── raw/                         ← Raw crawl (Google Places, SerpAPI)
│   ├── google_places/
│   ├── serpapi/
│   └── tourism_web/
├── processed/                   ← Enriched + cross-validated
│   ├── restaurant_enrichment.json
│   ├── accommodation_enrichment.json
│   └── attraction_enrichment.json
├── manual/                      ← Manual enrichment
├── scripts/                     ← Crawl scripts
└── audit/                       ← Validation + import logs

data_neo4j_v1/                   ← Schema V1 (initial)
data_neo4j_v2/                   ← Schema V2 (expanded)
data_neo4j_v3/                   ↑ Schema V3 (fixes)

dont_use/data_for_neo4j/         ← Raw source (giữ nguyên)
├── restaurant/restaurant_normalized.json
├── accommodation/data_v4.json
├── tourist_atractions/merged_tourist_attractions.json
├── specialty/dishes.json
├── event/festivals_processed.json
├── tour/quynhontourist/
└── agency/quanlyluhanh_agencies.json

graph_rag/data/                  ← GraphRAG runtime data
└── data_location.json
└── location_enrichment_*.json
└── community_summaries.json
```

---

## Các chỉ số chất lượng dữ liệu

| Tiêu chí | Trạng thái |
|---------|-----------|
| Cross-validation (Google vs SerpAPI) | ✅ 0 issues |
| Rating coverage (Restaurant) | 132/132 (100%) |
| Rating coverage (Accommodation) | 254/361 (70%) |
| Rating coverage (TouristAttraction) | 0/198 (cần bổ sung) |
| Opening hours coverage (Restaurant) | 98/132 (74%) |
| Opening hours coverage (Accommodation) | 7/361 (2%) |
| Ticket price coverage (TouristAttraction) | Một phần |
| Province mapping (all entities) | ✅ Hoàn tất |
| Dish location coverage | 48/151 (32%) |
