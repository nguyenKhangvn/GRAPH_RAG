import { RouteService } from "./routeService";

export class MapboxProxyRouteService extends RouteService {
  constructor({ apiBaseUrl } = {}) {
    super();
    this.endpoint = `${apiBaseUrl || "http://localhost:8000"}/api/mapbox/directions`;
  }

  async fetchRoute(locations, signal) {
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        points: locations.map((loc) => ({
          lat: loc.coordinates.lat,
          lng: loc.coordinates.lng,
        })),
        profile: "driving",
        overview: "full",
        geometries: "geojson",
      }),
      signal,
    });

    if (!response.ok) {
      throw new Error(`Routing API error ${response.status}`);
    }

    return response.json();
  }
}
