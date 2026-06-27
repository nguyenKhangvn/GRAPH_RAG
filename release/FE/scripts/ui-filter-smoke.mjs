import { filterLocations } from "../src/utils/locationFilters.js";

const sampleLocations = [
  {
    id: "a1",
    name: "Biển Hồ T'Nưng",
    type: "TouristAttraction",
    labels: ["TouristAttraction"],
  },
  {
    id: "a2",
    name: "Núi lửa Chư Đăng Ya",
    type: "TouristAttraction",
    labels: ["TouristAttraction"],
  },
  { id: "f1", name: "Phở khô Gia Lai", type: "Dish", labels: ["Dish"] },
  {
    id: "f2",
    name: "Nhà hàng Pleiku Garden",
    type: "Restaurant",
    labels: ["Restaurant"],
  },
  {
    id: "h1",
    name: "Khách sạn Mường Thanh Pleiku",
    type: "Accommodation",
    labels: ["Accommodation"],
  },
  { id: "c1", name: "Lễ hội cồng chiêng", type: "Event", labels: ["Event"] },
  { id: "c2", name: "Tour văn hóa bản địa", type: "Tour", labels: ["Tour"] },
];

const checks = [
  {
    name: "Chip all + keyword rỗng",
    category: "all",
    keyword: "",
    expected: 7,
  },
  {
    name: "Chip attraction + keyword bien",
    category: "attraction",
    keyword: "bien",
    expected: 1,
  },
  {
    name: "Chip food + keyword pho",
    category: "food",
    keyword: "pho",
    expected: 1,
  },
  {
    name: "Chip accommodation + keyword pleiku",
    category: "accommodation",
    keyword: "pleiku",
    expected: 1,
  },
  {
    name: "Chip culture + keyword cong chieng",
    category: "culture",
    keyword: "cong chieng",
    expected: 1,
  },
  {
    name: "Chip event + keyword le hoi",
    category: "event",
    keyword: "le hoi",
    expected: 1,
  },
  {
    name: "Chip food + keyword hotel (không khớp)",
    category: "food",
    keyword: "hotel",
    expected: 0,
  },
];

let passCount = 0;

for (const check of checks) {
  const result = filterLocations(
    sampleLocations,
    check.category,
    check.keyword,
  );
  const pass = result.length === check.expected;
  if (pass) {
    passCount += 1;
  }
  const status = pass ? "PASS" : "FAIL";
  console.log(
    `[${status}] ${check.name} | category=${check.category} | keyword="${check.keyword}" | expected=${check.expected} | actual=${result.length}`,
  );
}

console.log(`\nSummary: ${passCount}/${checks.length} checks passed.`);

if (passCount !== checks.length) {
  process.exit(1);
}
