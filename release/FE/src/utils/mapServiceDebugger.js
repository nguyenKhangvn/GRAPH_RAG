/**
 * Debug utility to catch and fix common map service errors
 * Ensures all method calls are safe and properly named
 */

const VALID_MAP_METHODS = [
  "initMap",
  "addMarker",
  "clearMarkers", // Note: NOT clearMarks
  "drawRoute",
  "fitBounds",
  "flyTo",
  "destroy",
];

const METHOD_FALLBACK_WARNED = new Set();

const warnOnce = (key, message) => {
  if (METHOD_FALLBACK_WARNED.has(key)) return;
  METHOD_FALLBACK_WARNED.add(key);
  console.warn(message);
};

/**
 * Wrap map service to catch invalid method calls
 */
export const createSafeMapService = (service) => {
  return new Proxy(service, {
    get(target, prop) {
      if (typeof prop !== "string") {
        return target[prop];
      }

      const mappedMethod = COMMON_TYPOS[prop] || prop;

      if (typeof target[mappedMethod] === "function") {
        return function (...args) {
          if (prop !== mappedMethod) {
            console.warn(
              `⚠️ Deprecated map method "${prop}" was called. Redirecting to "${mappedMethod}".`,
            );
          }

          if (
            !VALID_MAP_METHODS.includes(mappedMethod) &&
            mappedMethod !== "constructor"
          ) {
            console.warn(
              `⚠️ Invalid map method called: "${mappedMethod}". Expected one of: ${VALID_MAP_METHODS.join(", ")}.`,
            );
          }

          return target[mappedMethod](...args);
        };
      }

      if (VALID_MAP_METHODS.includes(mappedMethod)) {
        return function () {
          const key = `${target?.constructor?.name || "MapService"}:${mappedMethod}`;
          warnOnce(
            key,
            `⚠️ Map service method "${mappedMethod}" is missing. Skipping call safely.`,
          );
          return undefined;
        };
      }

      return target[prop];
    },
  });
};

/**
 * Common typos to fix in console
 */
export const COMMON_TYPOS = {
  clearMarks: "clearMarkers", // Most common - missing 'ers'
  removeMarkers: "clearMarkers", // Alternative
  clearMark: "clearMarkers", // Singular form
  addMarkers: "addMarker", // Plural version
};
