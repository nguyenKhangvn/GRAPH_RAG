import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapService } from "./mapService";

const DEFAULT_MAP_CENTER = [14.05, 108.35];
const DEFAULT_MAP_ZOOM = 8.6;

const escapeHtml = (value) =>
  String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

export class OSMService extends MapService {
  constructor({ onLoad } = {}) {
    super();
    this.onLoad = typeof onLoad === "function" ? onLoad : () => {};
    this.map = null;
    this.markerLayer = null;
    this.routeLayer = null;
    this.routeLayers = [];
  }

  initMap(container) {
    if (!container || this.map) return;

    this.map = L.map(container, {
      zoomControl: false,
      attributionControl: true,
    }).setView(DEFAULT_MAP_CENTER, DEFAULT_MAP_ZOOM);

    L.control.zoom({ position: "bottomright" }).addTo(this.map);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap",
      maxZoom: 19,
    }).addTo(this.map);

    this.markerLayer = L.layerGroup().addTo(this.map);
    this.onLoad();
  }

  addMarker(lat, lng, options = {}) {
    if (!this.map || !this.markerLayer) return;

    const iconColorStyle = options.color && !options.isActive ? `background-color: ${options.color}` : "";
    const icon = L.divIcon({
      className: options.isActive ? "leaflet-marker active" : "leaflet-marker",
      html: `<span style="${iconColorStyle}"></span>`,
      iconSize: [18, 18],
      iconAnchor: [9, 9],
    });

    const marker = L.marker([lat, lng], { icon }).bindPopup(
      `<div class="mapbox-popup-title">${escapeHtml(options.name || "")}</div>`,
    );

    marker.addTo(this.markerLayer);
  }

  clearMarkers() {
    if (this.markerLayer) {
      this.markerLayer.clearLayers();
    }
  }

  // Defensive alias for common typo: clearMarks -> clearMarkers
  clearMarks() {
    console.warn("⚠️ clearMarks() is deprecated. Use clearMarkers() instead.");
    return this.clearMarkers();
  }

  drawRoute(input) {
    if (!this.map) return;

    // Cleanup existing layers
    if (this.routeLayers && this.routeLayers.length > 0) {
      this.routeLayers.forEach((layer) => layer.remove());
    }
    this.routeLayers = [];

    if (this.routeLayer) {
      this.routeLayer.remove();
      this.routeLayer = null;
    }

    // Normalize input
    let routes = [];
    if (Array.isArray(input) && input.length > 0) {
      if (typeof input[0][0] === "number") {
        routes = [{ points: input, color: "#e2725b" }];
      } else {
        routes = input.filter(r => r && Array.isArray(r.points) && r.points.length > 1);
      }
    }

    routes.forEach((route) => {
      const layer = L.polyline(route.points, {
        color: route.color || "#e2725b",
        weight: 4,
        opacity: 0.9,
        lineJoin: "round",
      }).addTo(this.map);
      this.routeLayers.push(layer);
    });
  }

  fitBounds(locations) {
    if (!this.map) return;

    if (!Array.isArray(locations) || locations.length === 0) {
      this.map.flyTo(DEFAULT_MAP_CENTER, DEFAULT_MAP_ZOOM, {
        duration: 1.2,
      });
      return;
    }

    const bounds = L.latLngBounds(
      locations.map((loc) => [loc.coordinates.lat, loc.coordinates.lng]),
    );

    this.map.fitBounds(bounds, {
      paddingTopLeft: [window.innerWidth >= 768 ? 420 : 60, 110],
      paddingBottomRight: [60, 60],
      maxZoom: locations.length === 1 ? 14 : 12,
      animate: true,
      duration: 1.4,
    });
  }

  flyTo(lat, lng, zoom = 14) {
    if (!this.map) return;
    this.map.flyTo([lat, lng], zoom, {
      duration: 1,
    });
  }

  destroy() {
    if (this.map) {
      this.map.remove();
      this.map = null;
    }
    this.markerLayer = null;
    this.routeLayer = null;
  }
}
