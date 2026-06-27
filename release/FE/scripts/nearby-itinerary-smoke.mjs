const API_URL = process.env.API_URL || "http://127.0.0.1:8000/api/chat";
const TEST_QUERY =
  process.env.TEST_QUERY ||
  "Lập lịch trình 2 ngày ở Pleiku, ưu tiên các điểm du lịch gần nhau nhất để đỡ di chuyển";
const MAX_HOP_KM = Number(process.env.MAX_HOP_KM || 12);

const haversineKm = (lat1, lng1, lat2, lng2) => {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
};

const readMetadataFromSSE = async () => {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: TEST_QUERY, chat_history: [] }),
  });

  if (!response.ok || !response.body) {
    throw new Error(
      `API call failed: ${response.status} ${response.statusText}`,
    );
  }

  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (!payload || payload === "[DONE]") continue;
      let obj;
      try {
        obj = JSON.parse(payload);
      } catch {
        continue;
      }
      if (
        obj &&
        typeof obj === "object" &&
        Object.prototype.hasOwnProperty.call(obj, "intent") &&
        Array.isArray(obj.locations)
      ) {
        reader.cancel();
        return obj;
      }
    }
  }

  throw new Error("Metadata event not found in SSE stream");
};

try {
  const metadata = await readMetadataFromSSE();
  const locations = Array.isArray(metadata.locations) ? metadata.locations : [];

  console.log(`intent=${metadata.intent}`);
  console.log(`detected_location=${metadata.detected_location || ""}`);
  console.log(`locations_count=${locations.length}`);

  if (locations.length < 2) {
    throw new Error("Not enough locations to validate nearby itinerary");
  }

  let failures = 0;
  for (let i = 0; i < locations.length - 1; i += 1) {
    const a = locations[i]?.coordinates;
    const b = locations[i + 1]?.coordinates;
    if (!a || !b) continue;
    const km = haversineKm(
      Number(a.lat),
      Number(a.lng),
      Number(b.lat),
      Number(b.lng),
    );
    const ok = km <= MAX_HOP_KM;
    if (!ok) failures += 1;
    const status = ok ? "PASS" : "FAIL";
    console.log(
      `[${status}] hop ${i + 1}->${i + 2} ${locations[i].name} -> ${locations[i + 1].name} = ${km.toFixed(2)} km (limit ${MAX_HOP_KM} km)`,
    );
  }

  if (failures > 0) {
    throw new Error(
      `Nearby itinerary check failed with ${failures} hop(s) over limit`,
    );
  }

  console.log("Summary: PASS nearby itinerary guardrail.");
} catch (error) {
  console.error(`Summary: FAIL - ${error.message || String(error)}`);
  process.exit(1);
}
