import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict


class DirectionsService:
    """Provider-agnostic directions fetcher used as backend function-calling tool."""

    def __init__(self):
        self.provider = (os.getenv("DIRECTIONS_PROVIDER", "osrm") or "osrm").strip().lower()
        self.osrm_base_url = (
            os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org")
            or "https://router.project-osrm.org"
        ).rstrip("/")

    def get_directions(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        mode: str = "driving",
    ) -> Dict[str, Any]:
        """
        Function-calling style contract:
        get_directions(origin, destination, mode) -> normalized JSON payload.
        """
        if self.provider in {"osrm", "openstreetmap"}:
            return self._get_osrm_directions(origin, destination, mode)

        # Fallback to OSRM for unknown providers to keep system stable.
        return self._get_osrm_directions(origin, destination, mode)

    def build_external_map_url(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
    ) -> str:
        o_lat = float(origin["lat"])
        o_lng = float(origin["lng"])
        d_lat = float(destination["lat"])
        d_lng = float(destination["lng"])

        # Google Maps universal deep-link works well for FE and mobile handoff.
        return (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={urllib.parse.quote(f'{o_lat},{o_lng}') }"
            f"&destination={urllib.parse.quote(f'{d_lat},{d_lng}') }"
            "&travelmode=driving"
        )

    def build_external_map_url_flexible(
        self,
        *,
        origin_coords: Dict[str, Any] | None = None,
        origin_name: str | None = None,
        destination_node: Any | None = None,
        destination_name: str | None = None,
        travel_mode: str = "driving",
    ) -> str:
        """Build a Google Maps directions deep-link with flexible origin/destination.

        Handles mixed inputs:
        - origin: lat/lng dict (GPS) OR a place name string
        - destination: a resolved node with lat/lng OR just a name string

        Falls back gracefully when coordinates are unavailable.
        Returns an empty string when neither origin nor destination can be determined.
        """
        # --- Origin ---
        if origin_coords and "lat" in origin_coords and "lng" in origin_coords:
            o_param = f"{float(origin_coords['lat'])},{float(origin_coords['lng'])}"
        elif origin_name:
            o_param = origin_name
        else:
            return ""

        # --- Destination ---
        dest_lat = None
        dest_lng = None
        if destination_node is not None:
            meta = getattr(destination_node, "metadata", {}) or {}
            dest_lat = meta.get("lat")
            dest_lng = meta.get("lng")

        if dest_lat is not None and dest_lng is not None:
            d_param = f"{float(dest_lat)},{float(dest_lng)}"
        elif destination_name:
            d_param = destination_name
        else:
            return ""

        mode_map = {"walking": "walking", "walk": "walking", "foot": "walking"}
        travelmode = mode_map.get((travel_mode or "driving").strip().lower(), "driving")

        return (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={urllib.parse.quote(o_param)}"
            f"&destination={urllib.parse.quote(d_param)}"
            f"&travelmode={travelmode}"
        )

    def _get_osrm_directions(
        self,
        origin: Dict[str, Any],
        destination: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        o_lng = float(origin["lng"])
        o_lat = float(origin["lat"])
        d_lng = float(destination["lng"])
        d_lat = float(destination["lat"])

        profile = self._mode_to_osrm_profile(mode)
        coord = f"{o_lng},{o_lat};{d_lng},{d_lat}"
        url = (
            f"{self.osrm_base_url}/route/v1/{profile}/"
            f"{urllib.parse.quote(coord, safe=';,')}"
            "?overview=full&geometries=geojson"
        )

        with urllib.request.urlopen(url, timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))

        route = (payload.get("routes") or [None])[0]
        if not route:
            return {}

        coords = route.get("geometry", {}).get("coordinates", [])
        polyline = [
            {"lat": float(lat), "lng": float(lng)}
            for lng, lat in coords
            if isinstance(lat, (int, float)) and isinstance(lng, (int, float))
        ]

        return {
            "provider": "osrm",
            "travel_mode": mode,
            "road_distance_km": round(float(route.get("distance", 0.0)) / 1000.0, 2),
            "duration_min": int(round(float(route.get("duration", 0.0)) / 60.0)),
            "route_polyline": polyline,
        }

    def _mode_to_osrm_profile(self, mode: str) -> str:
        normalized = (mode or "driving").strip().lower()
        if normalized in {"walk", "walking", "foot"}:
            return "foot"
        return "driving"
