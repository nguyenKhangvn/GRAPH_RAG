import React from "react";
import {
  MapPin,
  Utensils,
  Hotel,
  Landmark,
  Trees,
  Calendar,
} from "lucide-react";

const categories = [
  { id: "all", label: "Tất cả", icon: <MapPin size={16} /> },
  { id: "attraction", label: "Địa điểm", icon: <Trees size={16} /> },
  { id: "food", label: "Ẩm thực", icon: <Utensils size={16} /> },
  { id: "accommodation", label: "Lưu trú", icon: <Hotel size={16} /> },
  { id: "culture", label: "Văn hóa", icon: <Landmark size={16} /> },
  { id: "event", label: "Sự kiện", icon: <Calendar size={16} /> },
];

const CategoryFilters = ({ activeCategory, onCategoryChange }) => (
  <div className="filter-scroll">
    {categories.map((cat) => (
      <button
        key={cat.id}
        className={`filter-btn ${activeCategory === cat.id ? "active" : ""}`}
        onClick={() => onCategoryChange(cat.id)}
      >
        {cat.icon}
        {cat.label}
      </button>
    ))}
  </div>
);

export default CategoryFilters;
