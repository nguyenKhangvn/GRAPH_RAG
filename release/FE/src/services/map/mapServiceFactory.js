import { MapboxService } from "./mapboxService";
import { OSMService } from "./osmService";
import { createSafeMapService } from "../../utils/mapServiceDebugger";

const warnedKeys = new Set();

const warnOnce = (key, message) => {
  if (warnedKeys.has(key)) return;
  warnedKeys.add(key);
  console.warn(message);
};

const ensureMethod = (service, methodName) => {
  if (typeof service[methodName] === "function") return;
  service[methodName] = () => {
    warnOnce(
      `missing:${service?.constructor?.name || "MapService"}:${methodName}`,
      `Map service is missing ${methodName}(). Call skipped safely.`,
    );
    return undefined;
  };
};

const normalizeMapServiceApi = (service) => {
  if (!service || typeof service !== "object") return service;

  // Guarantee clearMarkers exists.
  if (
    typeof service.clearMarkers !== "function" &&
    typeof service.clearMarks === "function"
  ) {
    service.clearMarkers = (...args) => service.clearMarks(...args);
  }

  // Guarantee clearMarks exists as a backward-compatible alias.
  if (
    typeof service.clearMarks !== "function" &&
    typeof service.clearMarkers === "function"
  ) {
    service.clearMarks = (...args) => service.clearMarkers(...args);
  }

  ensureMethod(service, "clearMarkers");
  ensureMethod(service, "clearMarks");
  return service;
};

export const resolveMapProvider = () => {
  const envValue =
    import.meta.env.VITE_MAP_PROVIDER || import.meta.env.VITE_MAP || "mapbox";
  const normalized = String(envValue).trim().toLowerCase();
  return normalized === "osm" ? "osm" : "mapbox";
};

export const createMapService = ({
  provider,
  mapboxAccessToken,
  mapboxStyleUrl,
  onLoad,
} = {}) => {
  const resolvedProvider = provider || resolveMapProvider();
  let service;

  if (resolvedProvider === "mapbox") {
    service = new MapboxService({
      accessToken: mapboxAccessToken,
      styleUrl: mapboxStyleUrl,
      onLoad,
    });
  } else {
    service = new OSMService({ onLoad });
  }

  return createSafeMapService(normalizeMapServiceApi(service));
};
