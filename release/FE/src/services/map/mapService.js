export class MapService {
  initMap() {
    throw new Error("MapService.initMap is not implemented");
  }

  addMarker() {
    throw new Error("MapService.addMarker is not implemented");
  }

  clearMarkers() {
    throw new Error("MapService.clearMarkers is not implemented");
  }

  drawRoute() {
    throw new Error("MapService.drawRoute is not implemented");
  }

  fitBounds() {
    throw new Error("MapService.fitBounds is not implemented");
  }

  flyTo() {
    throw new Error("MapService.flyTo is not implemented");
  }

  destroy() {
    throw new Error("MapService.destroy is not implemented");
  }

  // Defensive alias for common typo: clearMarks -> clearMarkers
  clearMarks() {
    console.warn("⚠️ clearMarks() is deprecated. Use clearMarkers() instead.");
    return this.clearMarkers();
  }
}
