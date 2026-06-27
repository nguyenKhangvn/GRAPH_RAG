const CATEGORY_LABEL_MAP = {
  attraction: new Set(["TouristAttraction"]),
  food: new Set(["Restaurant", "Dish"]),
  accommodation: new Set(["Accommodation"]),
  culture: new Set(["Event", "Tour", "Location"]),
  event: new Set(["Event"]),
};

export const normalizeText = (value) =>
  String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();

export const locationPrimaryType = (loc) => {
  if (loc?.type) return String(loc.type);
  if (Array.isArray(loc?.labels) && loc.labels.length > 0)
    return String(loc.labels[0]);
  return "Unknown";
};

export const locationMatchesCategory = (loc, category) => {
  if (!loc || category === "all") return true;
  const allowed = CATEGORY_LABEL_MAP[category];
  if (!allowed) return true;
  return allowed.has(locationPrimaryType(loc));
};

export const filterLocations = (locations, activeCategory, searchText) => {
  const source = Array.isArray(locations) ? locations : [];
  const query = normalizeText(searchText);

  return source.filter((loc) => {
    if (!locationMatchesCategory(loc, activeCategory)) {
      return false;
    }

    if (!query) {
      return true;
    }

    const haystack = normalizeText(
      [
        loc?.name,
        locationPrimaryType(loc),
        Array.isArray(loc?.labels) ? loc.labels.join(" ") : "",
      ].join(" "),
    );
    return haystack.includes(query);
  });
};
