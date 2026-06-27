import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { MapService } from "./mapService";

const DEFAULT_MAP_CENTER = [108.35, 14.05];
const DEFAULT_MAP_ZOOM = 8.6;
const ROUTE_SOURCE_ID = "route-line-source";
const ROUTE_LAYER_ID = "route-line-layer";

const escapeHtml = (value) =>
  String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

const toRouteGeoJson = (positions) => ({
  type: "Feature",
  properties: {},
  geometry: {
    type: "LineString",
    coordinates: positions.map(([lat, lng]) => [lng, lat]),
  },
});

export class MapboxService extends MapService {
  constructor({ accessToken, styleUrl, onLoad } = {}) {
    super();
    this.accessToken = accessToken || "";
    this.styleUrl = styleUrl || "mapbox://styles/mapbox/streets-v12";
    this.onLoad = typeof onLoad === "function" ? onLoad : () => {};
    this.map = null;
    this.markers = [];
    this.isLoaded = false;
  }

  initMap(container) {
    if (!container || this.map) return;
    if (!this.accessToken) {
      throw new Error("Missing VITE_MAPBOX_ACCESS_TOKEN for Mapbox provider");
    }

    mapboxgl.accessToken = this.accessToken;
    this.map = new mapboxgl.Map({
      container,
      style: this.styleUrl,
      center: DEFAULT_MAP_CENTER,
      zoom: DEFAULT_MAP_ZOOM,
      antialias: true,
      attributionControl: true,
    });

    this.map.addControl(
      new mapboxgl.NavigationControl({ showCompass: false }),
      "bottom-right",
    );

    this.map.on("load", () => {
      this.isLoaded = true;
      this.onLoad();
    });
  }

  addMarker(lat, lng, options = {}) {
    if (!this.map || !this.isLoaded) return;

    const markerEl = document.createElement("button");
    markerEl.type = "button";
    markerEl.className = "map-marker";
    markerEl.setAttribute(
      "aria-label",
      String(options.name || "Location marker"),
    );
    if (options.isActive) {
      markerEl.classList.add("active");
    } else if (options.color) {
      markerEl.style.backgroundColor = options.color;
    }

    const popup = new mapboxgl.Popup({ offset: 22 }).setHTML(
      `<div class="mapbox-popup-title">${escapeHtml(options.name || "")}</div>`,
    );

    const marker = new mapboxgl.Marker({
      element: markerEl,
      anchor: "bottom",
    })
      .setLngLat([lng, lat])
      .setPopup(popup)
      .addTo(this.map);

    this.markers.push(marker);
  }

  clearMarkers() {
    this.markers.forEach((marker) => marker.remove());
    this.markers = [];
  }

  // Defensive alias for common typo: clearMarks -> clearMarkers
  clearMarks() {
    console.warn("⚠️ clearMarks() is deprecated. Use clearMarkers() instead.");
    return this.clearMarkers();
  }

  drawRoute(input) {
    console.log("[MapboxService drawRoute] called with input:", input);
    if (!this.map || !this.isLoaded) return;

    // Normalize input to array of route objects: [{ points, color }]
    let routes = [];
    if (Array.isArray(input) && input.length > 0) {
      if (typeof input[0][0] === "number") {
        routes = [{ points: input, color: "#e2725b" }];
      } else {
        routes = input.filter(r => r && Array.isArray(r.points) && r.points.length > 1);
      }
    }

    // Cleanup existing daily and legacy layers/sources
    const style = this.map.getStyle();
    if (style && style.layers) {
      style.layers.forEach((layer) => {
        if (layer.id.startsWith("route-line-layer-") || layer.id === ROUTE_LAYER_ID) {
          try {
            this.map.removeLayer(layer.id);
          } catch (e) {
            console.warn("Failed to remove layer:", layer.id, e);
          }
        }
      });
    }

    if (style && style.sources) {
      Object.keys(style.sources).forEach((sourceId) => {
        if (sourceId.startsWith("route-line-source-") || sourceId === ROUTE_SOURCE_ID) {
          try {
            this.map.removeSource(sourceId);
          } catch (e) {
            console.warn("Failed to remove source:", sourceId, e);
          }
        }
      });
    }

    if (routes.length === 0) {
      return;
    }

    // Draw each route
    routes.forEach((route, index) => {
      const sourceId = `route-line-source-${index}`;
      const layerId = `route-line-layer-${index}`;
      const routeData = toRouteGeoJson(route.points);

      this.map.addSource(sourceId, {
        type: "geojson",
        data: routeData,
      });

      this.map.addLayer({
        id: layerId,
        type: "line",
        source: sourceId,
        layout: {
          "line-cap": "round",
          "line-join": "round",
        },
        paint: {
          "line-color": route.color || "#e2725b",
          "line-width": 4,
          "line-opacity": 0.9,
        },
      });
    });
  }

  fitBounds(locations) {
    if (!this.map || !this.isLoaded) return;

    if (!Array.isArray(locations) || locations.length === 0) {
      this.map.flyTo({
        center: DEFAULT_MAP_CENTER,
        zoom: DEFAULT_MAP_ZOOM,
        duration: 1200,
      });
      return;
    }

    const bounds = new mapboxgl.LngLatBounds();
    locations.forEach((loc) => {
      bounds.extend([loc.coordinates.lng, loc.coordinates.lat]);
    });

    this.map.fitBounds(bounds, {
      padding: {
        left: window.innerWidth >= 768 ? 420 : 60,
        right: 60,
        top: 110,
        bottom: 60,
      },
      maxZoom: locations.length === 1 ? 14 : 12,
      duration: 1400,
    });
  }

  flyTo(lat, lng, zoom = 14) {
    if (!this.map || !this.isLoaded) return;
    this.map.flyTo({
      center: [lng, lat],
      zoom,
      duration: 1000,
    });
  }

  destroy() {
    this.clearMarkers();
    if (this.map) {
      this.map.remove();
      this.map = null;
    }
    this.isLoaded = false;
  }
}
