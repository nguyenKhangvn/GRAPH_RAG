import { MapboxProxyRouteService } from "./mapboxProxyRouteService";
import { OSMRouteService } from "./osmRouteService";

export const createRouteService = ({ provider, apiBaseUrl } = {}) => {
  if (provider === "osm") {
    return new OSMRouteService();
  }
  return new MapboxProxyRouteService({ apiBaseUrl });
};
