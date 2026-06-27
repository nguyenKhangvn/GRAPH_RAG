import React, {
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Search, MapPin } from "lucide-react";
import CategoryFilters from "./CategoryFilters";
import {
  createMapService,
  resolveMapProvider,
} from "../services/map/mapServiceFactory";
import { createRouteService } from "../services/route/routeServiceFactory";

import { filterLocations } from "../utils/locationFilters";

const GRAPH_VIEW_PRELOADED_KEY = "graph_view_preloaded_once";
const MAP_PROVIDER_STORAGE_KEY = "runtime_map_provider";
const loadForceGraph = () => import("react-force-graph-2d");
const ForceGraph2D = lazy(loadForceGraph);

const MAPBOX_STYLE_URL =
  import.meta.env.VITE_MAPBOX_STYLE_URL || "mapbox://styles/mapbox/streets-v12";
const MAPBOX_ACCESS_TOKEN = import.meta.env.VITE_MAPBOX_ACCESS_TOKEN || "";
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const PROVIDER_LABEL = {
  mapbox: "Mapbox",
  osm: "OSM",
};

const formatRouteSummary = (meters, seconds) => {
  if (!meters || !seconds) return "";
  const km = (meters / 1000).toFixed(1);
  const mins = Math.round(seconds / 60);
  return `${km} km • ${mins} phút`;
};

const NODE_TYPE_COLORS = {
  TouristAttraction: "#2c5f2d",
  Restaurant: "#d9480f",
  Accommodation: "#1d4ed8",
  Event: "#9333ea",
  Tour: "#0f766e",
  Dish: "#b45309",
  TravelAgency: "#be185d",
  Location: "#334155",
  Unknown: "#64748b",
};

const DAY_COLORS = ["#2c5f2d", "#1d4ed8", "#7e22ce", "#b45309", "#0f766e"];

const getNodeType = (node) => {
  if (!node || !Array.isArray(node.labels) || node.labels.length === 0) {
    return "Unknown";
  }
  return node.labels[0] || "Unknown";
};

const getNodeColor = (node) =>
  NODE_TYPE_COLORS[getNodeType(node)] || NODE_TYPE_COLORS.Unknown;

const isSameLocation = (left, right) => {
  if (!left || !right) return false;
  return String(left.id) === String(right.id) || left.name === right.name;
};

const isValidCoordinate = (loc) =>
  Number.isFinite(loc?.coordinates?.lat) &&
  Number.isFinite(loc?.coordinates?.lng);

const safeClearMarkers = (service) => {
  if (!service) return;
  const clearMethod =
    typeof service.clearMarkers === "function"
      ? service.clearMarkers
      : typeof service.clearMarks === "function"
        ? service.clearMarks
        : null;

  if (!clearMethod) {
    console.warn(
      "Map service does not expose clearMarkers/clearMarks. Skip clear.",
    );
    return;
  }

  try {
    clearMethod.call(service);
  } catch (error) {
    console.warn(
      "Map service failed to clear markers. Continuing without crash.",
      error,
    );
  }
};

const resolveInitialProvider = () => {
  if (typeof window === "undefined") {
    return resolveMapProvider();
  }

  const stored = String(
    window.localStorage.getItem(MAP_PROVIDER_STORAGE_KEY) || "",
  )
    .trim()
    .toLowerCase();

  if (stored === "mapbox" || stored === "osm") {
    return stored;
  }

  return resolveMapProvider();
};

const MapProviderCanvas = ({
  mapProvider,
  filteredLocations,
  selectedLocation,
  routePositions,
  locationColors,
}) => {
  const containerRef = useRef(null);
  const serviceRef = useRef(null);
  const [isMapLoaded, setIsMapLoaded] = useState(false);

  useEffect(() => {
    if (!containerRef.current || serviceRef.current) return;

    const service = createMapService({
      provider: mapProvider,
      mapboxAccessToken: MAPBOX_ACCESS_TOKEN,
      mapboxStyleUrl: MAPBOX_STYLE_URL,
      onLoad: () => setIsMapLoaded(true),
    });

    try {
      service.initMap(containerRef.current);
      serviceRef.current = service;
    } catch (error) {
      console.error("Map init failed:", error);
    }

    return () => {
      service.destroy();
      serviceRef.current = null;
      setIsMapLoaded(false);
    };
  }, [mapProvider]);

  useEffect(() => {
    const service = serviceRef.current;
    if (!service || !isMapLoaded) return;

    safeClearMarkers(service);

    filteredLocations.filter(isValidCoordinate).forEach((loc) => {
      service.addMarker(loc.coordinates.lat, loc.coordinates.lng, {
        name: loc.name,
        isActive: isSameLocation(loc, selectedLocation),
        color: locationColors?.[loc.name] || null,
      });
    });
  }, [filteredLocations, selectedLocation, isMapLoaded, locationColors]);

  useEffect(() => {
    const service = serviceRef.current;
    if (!service || !isMapLoaded) return;

    const validLocations = filteredLocations.filter(isValidCoordinate);
    service.fitBounds(validLocations);
  }, [filteredLocations, isMapLoaded]);

  useEffect(() => {
    const service = serviceRef.current;
    if (!service || !isMapLoaded || !selectedLocation?.coordinates) return;

    service.flyTo(
      selectedLocation.coordinates.lat,
      selectedLocation.coordinates.lng,
      14,
    );
  }, [selectedLocation, isMapLoaded]);

  useEffect(() => {
    const service = serviceRef.current;
    if (!service || !isMapLoaded) return;

    service.drawRoute(routePositions);
  }, [routePositions, isMapLoaded]);

  if (mapProvider === "mapbox" && !MAPBOX_ACCESS_TOKEN) {
    return (
      <div className="mapbox-missing-token glass-panel">
        <h4>Chua cau hinh Mapbox token</h4>
        <p>
          Tao file .env trong FE va them VITE_MAPBOX_ACCESS_TOKEN de bat ban do
          Mapbox GL JS.
        </p>
      </div>
    );
  }

  return <div ref={containerRef} className="mapbox-canvas" />;
};

const normalizeString = (str) => {
  if (!str) return "";
  return str
    .toLowerCase()
    .replace(/[’‘'`]/g, "'") // replace curly quotes/apostrophes with straight single quote
    .replace(/\s+/g, " ") // collapse multiple spaces
    .trim();
};

const parseItineraryDaysFromText = (text) => {
  if (!text) return [];
  const dayPattern = /(?:^|\n)#{1,3}\s*(?:Ngày|Day|NGÀY)\s*(\d+)[^\n]*/gi;
  const days = [];
  let match;
  const positions = [];

  while ((match = dayPattern.exec(text)) !== null) {
    positions.push({ index: match.index, num: parseInt(match[1], 10), header: match[0].trim() });
  }

  if (positions.length === 0) return [];

  positions.forEach((pos, i) => {
    const start = pos.index + pos.header.length;
    const end = i + 1 < positions.length ? positions[i + 1].index : text.length;
    const chunk = text.slice(start, end).trim();

    const slots = [];
    const timeLinePattern = /[\*\-]?\s*(\d{1,2}:\d{2}(?:\s*[-–—]\s*\d{1,2}:\d{2})?):?\s*(.+)/g;
    let slotMatch;
    while ((slotMatch = timeLinePattern.exec(chunk)) !== null) {
      const time = slotMatch[1].trim();
      const place = slotMatch[2].replace(/^\*+|\*+$/g, "").trim();
      if (place) slots.push({ time, place });
    }

    if (slots.length === 0) {
      const bulletPattern = /^[\*\-]\s+(.+)/gm;
      let bMatch;
      while ((bMatch = bulletPattern.exec(chunk)) !== null) {
        const line = bMatch[1].replace(/^\*+|\*+$/g, "").trim();
        if (line) slots.push({ time: "", place: line });
      }
    }

    days.push({
      dayNum: pos.num,
      slots,
    });
  });

  return days;
};

const MapInterface = ({
  mapLocations,
  mapIntent,
  mapGraph,
  mapDistance,
  mapSafety,
  mapConstraintWarning,
  mapDailyPlan,
  routeSummary,
  setRouteSummary,
  lastItineraryText,
}) => {
  const [activeCategory, setActiveCategory] = useState("all");
  const [searchText, setSearchText] = useState("");
  const [activeView, setActiveView] = useState("map");
  const [mapProvider, setMapProvider] = useState(resolveInitialProvider);
  const [routePositions, setRoutePositions] = useState([]);
  const [selectedGraphNode, setSelectedGraphNode] = useState(null);
  const [isGraphBootstrapping, setIsGraphBootstrapping] = useState(false);
  const routeCacheRef = useRef(new Map());

  const routeService = useMemo(
    () =>
      createRouteService({ provider: mapProvider, apiBaseUrl: API_BASE_URL }),
    [mapProvider],
  );

  const isGraphViewEnabled =
    String(import.meta.env.VITE_ENABLE_GRAPH_VIEW || "true").toLowerCase() ===
    "true";

  const normalizedIntent = String(mapIntent || "").toUpperCase();
  const isItineraryIntent = normalizedIntent === "TOUR_PLAN";
  const isDistanceIntent = normalizedIntent === "DISTANCE_QUERY";

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(MAP_PROVIDER_STORAGE_KEY, mapProvider);
  }, [mapProvider]);

  useEffect(() => {
    routeCacheRef.current.clear();
  }, [mapProvider]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const hasOpenedGraphBefore =
      window.sessionStorage.getItem(GRAPH_VIEW_PRELOADED_KEY) === "1";
    if (!hasOpenedGraphBefore) return;

    const idleCb =
      window.requestIdleCallback || ((cb) => window.setTimeout(cb, 1200));

    const idleId = idleCb(() => {
      loadForceGraph().catch(() => {
        // Keep map view stable even if preload fails.
      });
    });

    return () => {
      if (window.cancelIdleCallback && typeof idleId === "number") {
        window.cancelIdleCallback(idleId);
      } else {
        window.clearTimeout(idleId);
      }
    };
  }, []);

  const handleOpenGraphView = async () => {
    setActiveView("graph");
    if (typeof window === "undefined") return;

    const wasOpenedBefore =
      window.sessionStorage.getItem(GRAPH_VIEW_PRELOADED_KEY) === "1";
    if (wasOpenedBefore) return;

    setIsGraphBootstrapping(true);
    window.sessionStorage.setItem(GRAPH_VIEW_PRELOADED_KEY, "1");
    try {
      await loadForceGraph();
    } finally {
      setIsGraphBootstrapping(false);
    }
  };

  const filteredLocations = useMemo(() => {
    return filterLocations(mapLocations, activeCategory, searchText);
  }, [mapLocations, activeCategory, searchText]);

  const shouldDrawRoute =
    (isItineraryIntent || isDistanceIntent) &&
    filteredLocations &&
    filteredLocations.length > 1;

  const graphData = useMemo(() => {
    if (
      mapGraph &&
      Array.isArray(mapGraph.nodes) &&
      Array.isArray(mapGraph.links)
    ) {
      return mapGraph;
    }
    return { nodes: [], links: [] };
  }, [mapGraph]);

  const selectedLocation = useMemo(() => {
    if (!selectedGraphNode || !Array.isArray(mapLocations)) return null;
    return (
      mapLocations.find(
        (loc) => String(loc.id) === String(selectedGraphNode.id),
      ) ||
      mapLocations.find((loc) => loc.name === selectedGraphNode.name) ||
      null
    );
  }, [selectedGraphNode, mapLocations]);

  const graphLegend = useMemo(() => {
    if (!graphData?.nodes?.length) return [];
    const counts = {};
    graphData.nodes.forEach((node) => {
      const type = getNodeType(node);
      counts[type] = (counts[type] || 0) + 1;
    });
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([type, count]) => ({
        type,
        count,
        color: NODE_TYPE_COLORS[type] || NODE_TYPE_COLORS.Unknown,
      }));
  }, [graphData]);

  // Build a location → day number map from parsed text or daily cluster plan
  const locationDayMap = useMemo(() => {
    const dayMap = {};
    if (isItineraryIntent && lastItineraryText && Array.isArray(mapLocations)) {
      const days = parseItineraryDaysFromText(lastItineraryText);
      if (days.length > 0) {
        mapLocations.forEach((loc) => {
          const normLocName = normalizeString(loc.name);
          for (const day of days) {
            for (const slot of day.slots) {
              const normSlotPlace = normalizeString(slot.place);
              if (
                normSlotPlace.includes(normLocName) ||
                normLocName.includes(normSlotPlace)
              ) {
                dayMap[loc.name] = day.dayNum;
                break;
              }
            }
            if (dayMap[loc.name]) break;
          }
        });
      }
    }

    // Fallback to mapDailyPlan for any remaining locations
    if (isItineraryIntent && Array.isArray(mapDailyPlan)) {
      mapDailyPlan.forEach((dayEntry) => {
        const dayNum = dayEntry?.day;
        const pointNames = dayEntry?.point_names || [];
        pointNames.forEach((name) => {
          if (name && dayNum && !dayMap[name]) {
            dayMap[name] = dayNum;
          }
        });
      });
    }
    return dayMap;
  }, [isItineraryIntent, lastItineraryText, mapLocations, mapDailyPlan]);

  const locationColors = useMemo(() => {
    const colors = {};
    Object.entries(locationDayMap).forEach(([name, dayNum]) => {
      const colorIndex = (dayNum - 1) % DAY_COLORS.length;
      colors[name] = DAY_COLORS[colorIndex];
    });
    return colors;
  }, [locationDayMap]);

  useEffect(() => {
    console.log("[DEBUG Routing useEffect] Triggered", {
      shouldDrawRoute,
      filteredLocationsLength: filteredLocations?.length,
      isDistanceIntent,
      mapProvider,
      mapIntent,
      isItineraryIntent
    });

    if (!shouldDrawRoute) {
      console.log("[DEBUG Routing useEffect] shouldDrawRoute is false, resetting routePositions");
      setRoutePositions([]);
      setRouteSummary("");
      return;
    }

    const validLocations = filteredLocations.filter(
      (loc) =>
        loc?.coordinates &&
        Number.isFinite(loc.coordinates.lat) &&
        Number.isFinite(loc.coordinates.lng),
    );

    console.log("[DEBUG Routing useEffect] validLocations", validLocations);

    if (validLocations.length < 2) {
      console.log("[DEBUG Routing useEffect] validLocations.length < 2, resetting");
      setRoutePositions([]);
      setRouteSummary("");
      return;
    }

    const controller = new AbortController();

    if (
      isDistanceIntent &&
      mapDistance &&
      Array.isArray(mapDistance.route_polyline) &&
      mapDistance.route_polyline.length > 1
    ) {
      console.log("[DEBUG Routing useEffect] Using distance intent route polyline:", mapDistance.route_polyline);
      const positions = mapDistance.route_polyline.map((p) => [p.lat, p.lng]);
      setRoutePositions([{ points: positions, color: "#e2725b" }]);
      if (
        Number.isFinite(mapDistance.road_distance_km) &&
        Number.isFinite(mapDistance.duration_min)
      ) {
        setRouteSummary(
          `${mapDistance.road_distance_km} km • ${mapDistance.duration_min} phút`,
        );
      } else if (Number.isFinite(mapDistance.straight_distance_km)) {
        setRouteSummary(
          `${mapDistance.straight_distance_km} km (đường chim bay)`,
        );
      } else {
        setRouteSummary("");
      }
      return () => controller.abort();
    }

    const fetchRoadRoute = async () => {
      console.log("[DEBUG Routing fetchRoadRoute] Started");
      try {
        if (isItineraryIntent) {
          const days = [...new Set(Object.values(locationDayMap))].sort((a, b) => a - b);
          if (days.length === 0) {
            // Fallback: fetch a single route
            const routeKey = `${mapProvider}:${validLocations
              .map(
                (loc) =>
                  `${loc.coordinates.lat.toFixed(6)},${loc.coordinates.lng.toFixed(
                    6,
                  )}`,
              )
              .join("|")}`;

            const cachedRoute = routeCacheRef.current.get(routeKey);
            if (cachedRoute) {
              setRoutePositions([{ points: cachedRoute.positions, color: "#e2725b" }]);
              setRouteSummary(cachedRoute.summary);
              return;
            }

            const data = await routeService.fetchRoute(
              validLocations,
              controller.signal,
            );
            const route = data?.routes?.[0];
            const geometry = route?.geometry?.coordinates;
            if (!Array.isArray(geometry) || geometry.length < 2) {
              throw new Error("Routing API did not return valid geometry");
            }
            const positions = geometry.map(([lng, lat]) => [lat, lng]);
            setRoutePositions([{ points: positions, color: "#e2725b" }]);
            const summary = formatRouteSummary(route.distance, route.duration);
            setRouteSummary(summary);
            routeCacheRef.current.set(routeKey, { positions, summary });
            return;
          }

          // Fetch routing for each day in parallel
          const routePromises = days.map(async (dayNum) => {
            const dayLocations = validLocations.filter(
              (loc) => locationDayMap[loc.name] === dayNum
            );
            if (dayLocations.length < 2) return null;

            const routeKey = `${mapProvider}:${dayLocations
              .map(
                (loc) =>
                  `${loc.coordinates.lat.toFixed(6)},${loc.coordinates.lng.toFixed(
                    6,
                  )}`,
              )
              .join("|")}`;

            const cachedRoute = routeCacheRef.current.get(routeKey);
            if (cachedRoute) {
              return { dayNum, positions: cachedRoute.positions, summary: cachedRoute.summary };
            }

            try {
              const data = await routeService.fetchRoute(
                dayLocations,
                controller.signal,
              );
              const route = data?.routes?.[0];
              const geometry = route?.geometry?.coordinates;
              if (!Array.isArray(geometry) || geometry.length < 2) {
                throw new Error(`Routing API failed for Day ${dayNum}`);
              }
              const positions = geometry.map(([lng, lat]) => [lat, lng]);
              const summary = formatRouteSummary(route.distance, route.duration);
              routeCacheRef.current.set(routeKey, { positions, summary });
              return { dayNum, positions, summary };
            } catch (err) {
              console.warn(`Road route failed for Day ${dayNum}, using straight line fallback:`, err);
              const fallbackPositions = dayLocations.map((loc) => [
                loc.coordinates.lat,
                loc.coordinates.lng,
              ]);
              return { dayNum, positions: fallbackPositions, summary: "" };
            }
          });

          const results = await Promise.all(routePromises);
          const dailyRoutes = results
            .filter(Boolean)
            .map((res) => ({
              points: res.positions,
              color: DAY_COLORS[(res.dayNum - 1) % DAY_COLORS.length],
            }));
          setRoutePositions(dailyRoutes);
          return;
        }

        // Single route fallback (e.g. non-itinerary)
        const routeKey = `${mapProvider}:${validLocations
          .map(
            (loc) =>
              `${loc.coordinates.lat.toFixed(6)},${loc.coordinates.lng.toFixed(
                6,
              )}`,
          )
          .join("|")}`;

        const cachedRoute = routeCacheRef.current.get(routeKey);
        if (cachedRoute) {
          setRoutePositions([{ points: cachedRoute.positions, color: "#e2725b" }]);
          setRouteSummary(cachedRoute.summary);
          return;
        }

        const data = await routeService.fetchRoute(
          validLocations,
          controller.signal,
        );
        const route = data?.routes?.[0];
        const geometry = route?.geometry?.coordinates;
        if (!Array.isArray(geometry) || geometry.length < 2) {
          throw new Error("Routing API did not return valid geometry");
        }
        const positions = geometry.map(([lng, lat]) => [lat, lng]);
        setRoutePositions([{ points: positions, color: "#e2725b" }]);
        const summary = formatRouteSummary(route.distance, route.duration);
        setRouteSummary(summary);
        routeCacheRef.current.set(routeKey, { positions, summary });
      } catch (error) {
        if (error.name === "AbortError") return;
        console.error("Failed to fetch route, fallback to straight line:", error);
        const fallbackPositions = validLocations.map((loc) => [
          loc.coordinates.lat,
          loc.coordinates.lng,
        ]);
        setRoutePositions([{ points: fallbackPositions, color: "#e2725b" }]);
        setRouteSummary("");
      }
    };

    const debounceId = window.setTimeout(() => {
      fetchRoadRoute();
    }, 500);

    return () => {
      window.clearTimeout(debounceId);
      controller.abort();
    };
  }, [
    filteredLocations,
    shouldDrawRoute,
    isDistanceIntent,
    mapDistance,
    mapProvider,
    routeService,
    locationDayMap,
  ]);

  const providerBadge = PROVIDER_LABEL[mapProvider] || "Mapbox";
  const distanceMapUrl =
    isDistanceIntent && typeof mapDistance?.map_url === "string"
      ? mapDistance.map_url
      : "";
  const itinerarySafetyWarning =
    isItineraryIntent && mapSafety?.safe_for_fe === false
      ? String(
          mapSafety.warning ||
            "Lich trinh co mot so chang di chuyen dai, vui long kiem tra truoc khi di.",
        )
      : "";

  return (
    <div
      className={`map-container ${activeView === "graph" ? "graph-mode" : ""}`}
      style={{ position: "relative" }}
    >
      <div className="map-overlay-top">
        <div className="provider-toolbar glass-panel">
          <span className="provider-badge" title="Map provider đang chạy">
            Lựa chọn: {providerBadge}
          </span>
          <div
            className="provider-toggle"
            role="group"
            aria-label="Map provider switch"
          >
            <button
              type="button"
              className={`view-toggle-btn ${mapProvider === "mapbox" ? "active" : ""}`}
              onClick={() => setMapProvider("mapbox")}
            >
              Mapbox
            </button>
            <button
              type="button"
              className={`view-toggle-btn ${mapProvider === "osm" ? "active" : ""}`}
              onClick={() => setMapProvider("osm")}
            >
              OSM (free)
            </button>
          </div>
        </div>

        {isGraphViewEnabled && (
          <div className="view-toggle glass-panel">
            <button
              type="button"
              className={`view-toggle-btn ${activeView === "map" ? "active" : ""}`}
              onClick={() => setActiveView("map")}
            >
              Map View
            </button>
            {/* <button
              type="button"
              className={`view-toggle-btn ${activeView === "graph" ? "active" : ""}`}
              onClick={handleOpenGraphView}
            >
              {isGraphBootstrapping ? "Graph View (loading...)" : "Graph View"}
            </button> */}
          </div>
        )}

        <div className="search-bar glass-panel">
          <Search size={20} color="var(--text-muted)" />
          <input
            type="text"
            className="search-input"
            placeholder="Tìm địa điểm..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
        </div>
        <CategoryFilters
          activeCategory={activeCategory}
          onCategoryChange={setActiveCategory}
        />
      </div>

      {activeView === "map" ? (
        <MapProviderCanvas
          mapProvider={mapProvider}
          filteredLocations={filteredLocations}
          selectedLocation={selectedLocation}
          routePositions={shouldDrawRoute ? routePositions : []}
          locationColors={locationColors}
        />
      ) : (
        <div className="graph-view-canvas">
          <div className="graph-view-center">GraphRAG Explainability</div>
          {graphLegend.length > 0 && (
            <div className="graph-legend glass-panel">
              {graphLegend.map((item) => (
                <div key={item.type} className="graph-legend-item">
                  <span
                    className="graph-legend-dot"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="graph-legend-text">
                    {item.type} ({item.count})
                  </span>
                </div>
              ))}
            </div>
          )}
          <div className="graph-view-stage">
            {graphData.nodes.length > 0 ? (
              <Suspense
                fallback={
                  <div className="graph-view-empty">Đang tải Graph View...</div>
                }
              >
                <ForceGraph2D
                  graphData={graphData}
                  nodeLabel={(node) => {
                    const labelText =
                      Array.isArray(node.labels) && node.labels.length > 0
                        ? `${node.name} (${node.labels.join(", ")})`
                        : node.name;
                    return labelText;
                  }}
                  linkLabel={(link) => link.relation || "RELATED"}
                  linkDirectionalArrowLength={5}
                  linkDirectionalArrowRelPos={1}
                  linkCurvature={0.08}
                  nodeRelSize={5}
                  nodeColor={getNodeColor}
                  cooldownTicks={120}
                  onNodeClick={(node) => {
                    setSelectedGraphNode({ id: node.id, name: node.name });
                    setActiveView("map");
                  }}
                />
              </Suspense>
            ) : (
              <div className="graph-view-empty">
                Chưa có dữ liệu quan hệ để hiển thị Graph View.
              </div>
            )}
          </div>
        </div>
      )}

      {filteredLocations && filteredLocations.length > 0 && (
        <div className="location-panel glass-panel">
          <div className="location-panel-header">
            <MapPin size={14} color="var(--brand-forest-green)" />
            <span>
              {filteredLocations.length}
              {Array.isArray(mapLocations) ? `/${mapLocations.length}` : ""} địa
              điểm
              {searchText ? " (đã lọc)" : " được tìm thấy"}
              {mapIntent ? ` • ${mapIntent}` : ""}
              {` • ${providerBadge}`}
              {(isItineraryIntent || isDistanceIntent) && routeSummary
                ? ` • ${routeSummary}`
                : ""}
            </span>
          </div>
          {distanceMapUrl ? (
            <div className="location-panel-map-link">
              <a href={distanceMapUrl} target="_blank" rel="noreferrer">
                Mở bản đồ chỉ đường chi tiết
              </a>
            </div>
          ) : null}
          {itinerarySafetyWarning ? (
            <div
              className="location-panel-warning"
              role="status"
              aria-live="polite"
            >
              {itinerarySafetyWarning}
            </div>
          ) : null}
          {mapConstraintWarning && isItineraryIntent && (
            <div className="location-panel-constraint-warning" role="alert">
              <span>⚠️</span>
              <span>{mapConstraintWarning.message}</span>
            </div>
          )}
          <div className="location-panel-list">
            {filteredLocations.map((loc, i) => {
              const dayNum = locationDayMap[loc.name];
              const isActive = isSameLocation(loc, selectedLocation);
              return (
                <div
                  key={loc.id || i}
                  className={`location-item ${isActive ? "active" : ""}`}
                  onClick={() => setSelectedGraphNode({ id: loc.id, name: loc.name })}
                  style={{ cursor: "pointer" }}
                >
                  <span className="location-num">{i + 1}</span>
                  {dayNum && isItineraryIntent && (
                    <span className="location-item-day">N{dayNum}</span>
                  )}
                  <span className="location-name">{loc.name}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default MapInterface;
