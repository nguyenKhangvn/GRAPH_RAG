import { RouteService } from "./routeService";

const OSRM_PUBLIC_ROUTE_URL =
  "https://router.project-osrm.org/route/v1/driving";

const formatOsmRouteUrl = (locations) => {
  const coords = locations
    .map((loc) => `${loc.coordinates.lng},${loc.coordinates.lat}`)
    .join(";");
  return `${OSRM_PUBLIC_ROUTE_URL}/${coords}?overview=full&geometries=geojson`;
};

export class OSMRouteService extends RouteService {
  async fetchRoute(locations, signal) {
    const response = await fetch(formatOsmRouteUrl(locations), {
      method: "GET",
      signal,
    });

    if (!response.ok) {
      throw new Error(`OSRM API error ${response.status}`);
    }

    return response.json();
  }
}
