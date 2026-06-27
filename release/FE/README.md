# FE - Gia Lai Travel AI

Frontend su dung React + Vite va ho tro 2 map provider:

- `mapbox`: provider mac dinh (giao dien dep, can token)
- `osm`: provider free 100% cho demo/test (Leaflet + OpenStreetMap)

## Cai dat

1. Cai dependencies:

```bash
npm install
```

2. Tao file `.env` trong thu muc `FE`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_MAP_PROVIDER=mapbox
VITE_MAPBOX_ACCESS_TOKEN=your_mapbox_access_token
VITE_MAPBOX_STYLE_URL=mapbox://styles/mapbox/streets-v12
VITE_ENABLE_GRAPH_VIEW=true
```

## Chay dev

```bash
npm run dev
```

## Build production

```bash
npm run build
```

## Ghi chu Mapbox

- Bat buoc co `VITE_MAPBOX_ACCESS_TOKEN` de hien thi map nen (map tiles/style).
- Co the thay doi giao dien ban do bang `VITE_MAPBOX_STYLE_URL`.
- Neu chua cau hinh token, UI se hien thong bao huong dan ngay tren map panel.

## Ghi chu OSM (free cho demo)

- Set `VITE_MAP_PROVIDER=osm` de bat OpenStreetMap + Leaflet.
- OSM mode khong can API key.
- Routing trong OSM mode dung OSRM public endpoint:
  - `https://router.project-osrm.org/route/v1/driving/...`
- Phu hop de test local, demo nhanh, tiet kiem quota Mapbox.

## Switch provider

Factory map service se tu dong chon provider theo env:

- `VITE_MAP_PROVIDER=mapbox` -> dung `MapboxService`
- `VITE_MAP_PROVIDER=osm` -> dung `OSMService`

## 🎯 Nguyen tac cot loi (bao ve quota)

- KHONG goi Mapbox Directions truc tiep tu frontend.
- Moi request route phai di qua backend endpoint `POST /api/mapbox/directions`.
- Frontend phai debounce truoc khi goi route API (hien tai da debounce 500ms).
- Backend bat buoc ratelimit + internal quota + cache.

Kien truc:

Frontend (React)
-> Backend (FastAPI)
-> Mapbox API

### Vi sao khong goi truc tiep tu frontend?

- De lo API key va tang rui ro bi abuse quota.
- Khong kiem soat duoc tan suat request theo user.
- Khong co diem gom de cache route va monitor usage.
