import {
  filterLocations,
  locationMatchesCategory,
  normalizeText,
} from "../src/utils/locationFilters.js";

const API_URL = process.env.API_URL || "http://127.0.0.1:8000/api/chat";
const TEST_QUERY =
  process.env.TEST_QUERY ||
  "du lịch 3 ngày 2 đêm tại Gia Lai ngân sách 5 triệu cho 2 người, đi từ Pleiku";

const CATEGORIES = [
  "all",
  "attraction",
  "food",
  "accommodation",
  "culture",
  "event",
];

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

const pickKeyword = (locations) => {
  for (const loc of locations) {
    const tokens = String(loc?.name || "")
      .split(/\s+/)
      .map((t) => normalizeText(t))
      .filter((t) => t.length >= 3 && /^[a-z0-9]+$/.test(t));
    if (tokens.length > 0) {
      return { keyword: tokens[0], sourceName: loc.name };
    }
  }
  return null;
};

const checks = [];
let pass = 0;
let fail = 0;
let skip = 0;

const logResult = (status, title, detail) => {
  checks.push({ status, title, detail });
  if (status === "PASS") pass += 1;
  if (status === "FAIL") fail += 1;
  if (status === "SKIP") skip += 1;
  console.log(`[${status}] ${title} | ${detail}`);
};

try {
  const metadata = await readMetadataFromSSE();
  const locations = Array.isArray(metadata.locations) ? metadata.locations : [];

  logResult(
    locations.length > 0 ? "PASS" : "FAIL",
    "Metadata có locations",
    `count=${locations.length}`,
  );

  const hasTypeOrLabels = locations.every(
    (loc) =>
      Boolean(loc?.type) ||
      (Array.isArray(loc?.labels) && loc.labels.length > 0),
  );
  logResult(
    hasTypeOrLabels ? "PASS" : "FAIL",
    "Mỗi location có type/labels",
    hasTypeOrLabels ? "ok" : "missing type/labels",
  );

  for (const category of CATEGORIES) {
    const expected =
      category === "all"
        ? locations.length
        : locations.filter((loc) => locationMatchesCategory(loc, category))
            .length;
    const actual = filterLocations(locations, category, "").length;
    const status = expected === actual ? "PASS" : "FAIL";
    logResult(
      status,
      `Chip ${category} lọc đúng`,
      `expected=${expected}, actual=${actual}`,
    );
  }

  const keywordPick = pickKeyword(locations);
  if (!keywordPick) {
    logResult(
      "SKIP",
      "Keyword search từ dữ liệu thật",
      "không chọn được keyword hợp lệ",
    );
  } else {
    const keywordMatches = filterLocations(
      locations,
      "all",
      keywordPick.keyword,
    );
    const status = keywordMatches.length > 0 ? "PASS" : "FAIL";
    logResult(
      status,
      "Keyword search có kết quả",
      `keyword=${keywordPick.keyword}, source=${keywordPick.sourceName}, matches=${keywordMatches.length}`,
    );
  }
} catch (error) {
  logResult(
    "FAIL",
    "Smoke test gọi API metadata thật",
    error.message || String(error),
  );
}

console.log(
  `\nSummary: PASS=${pass}, FAIL=${fail}, SKIP=${skip}, TOTAL=${checks.length}`,
);
if (fail > 0) {
  process.exit(1);
}
