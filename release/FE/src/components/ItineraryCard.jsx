import React, { useState } from "react";
import { ChevronDown, ChevronUp, Clock, MapPin, Moon, AlertTriangle, Hotel, Car } from "lucide-react";

// ---------------------------------------------------------------------------
// Helpers: parse markdown itinerary text into structured data
// ---------------------------------------------------------------------------

/**
 * Parse AI-generated itinerary markdown into an array of day objects.
 * Handles both "Ngày X" and "Day X" patterns.
 *
 * @returns {Array<{dayNum: number, label: string, region: string, slots: Array<{time: string, place: string}>}>}
 */
function parseItineraryDays(text) {
  if (!text) return [];

  // Split by day headers: "Ngày 1", "Ngày 2", "Day 1", etc.
  const dayPattern = /(?:^|\n).{0,10}?(?:Ngày|Day|NGÀY)\s*(\d+)[^\n]*/gi;
  const days = [];
  let match;
  const positions = [];

  while ((match = dayPattern.exec(text)) !== null) {
    positions.push({ index: match.index, num: parseInt(match[1], 10), header: match[0].trim() });
  }

  if (positions.length === 0) return [];

  positions.forEach((pos, i) => {
    const start = pos.index + pos.header.length;
    const end = i + 1 < positions.length ? positions[i + 1].index : text.length;
    const chunk = text.slice(start, end).trim();

    // Parse time slots: lines starting with time pattern "08:00", "HH:MM"
    const slots = [];
    const timeLinePattern = /[\*\-]?\s*(\d{1,2}:\d{2}(?:\s*[-–—]\s*\d{1,2}:\d{2})?):?\s*(.+)/g;
    let slotMatch;
    while ((slotMatch = timeLinePattern.exec(chunk)) !== null) {
      const time = slotMatch[1].trim();
      const place = slotMatch[2].replace(/^\*+|\*+$/g, "").trim(); // strip bold markers
      if (place) slots.push({ time, place });
    }

    // If no time-based slots, try bullet points as places
    if (slots.length === 0) {
      const bulletPattern = /^[\*\-]\s+(.+)/gm;
      let bMatch;
      while ((bMatch = bulletPattern.exec(chunk)) !== null) {
        const line = bMatch[1].replace(/^\*+|\*+$/g, "").trim();
        if (line) slots.push({ time: "", place: line });
      }
    }

    // Extract region label: "📍 Khu vực: ..." or "**Khu vực:**"
    const regionMatch = chunk.match(/(?:📍\s*)?(?:Khu\s+vực|Khu vuc)[:\s]+([^\n]+)/i);
    const region = regionMatch ? regionMatch[1].replace(/\*+/g, "").trim() : "";

    days.push({
      dayNum: pos.num,
      label: `Ngày ${pos.num}`,
      region,
      slots,
    });
  });

  return days;
}

/**
 * Parse lodging suggestions from text: "Đêm X: ..." or "Khách sạn..." lines
 */
function parseLodging(text) {
  if (!text) return [];
  const lodgingPattern = /(?:Đêm|Dem|Nghỉ đêm|Nghi dem)\s*\d*[:\s]+([^\n]+)/gi;
  const results = [];
  let m;
  while ((m = lodgingPattern.exec(text)) !== null) {
    const raw = m[1].replace(/\*+/g, "").trim();
    if (raw) results.push(raw);
  }
  return results;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const DAY_GRADIENTS = [
  "linear-gradient(135deg, #2c5f2d 0%, #4a9e4b 100%)",
  "linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%)",
  "linear-gradient(135deg, #7e22ce 0%, #a855f7 100%)",
  "linear-gradient(135deg, #b45309 0%, #d97706 100%)",
  "linear-gradient(135deg, #0f766e 0%, #14b8a6 100%)",
];

function DayCard({ day, index, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  const gradient = DAY_GRADIENTS[index % DAY_GRADIENTS.length];

  return (
    <div className="itin-day-card">
      {/* Day header */}
      <button
        className="itin-day-header"
        style={{ background: gradient }}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <div className="itin-day-header-left">
          <span className="itin-day-badge">{day.label}</span>
          {day.region && (
            <span className="itin-day-region">
              <MapPin size={11} />
              {day.region}
            </span>
          )}
        </div>
        <div className="itin-day-chevron">
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {/* Slots timeline */}
      {open && (
        <div className="itin-slots">
          {day.slots.length > 0 ? (
            day.slots.map((slot, si) => (
              <div key={si} className="itin-slot">
                <div className="itin-slot-left">
                  <div className="itin-slot-dot" />
                  {si < day.slots.length - 1 && <div className="itin-slot-line" />}
                </div>
                <div className="itin-slot-content">
                  {slot.time && (
                    <div className="itin-slot-time">
                      <Clock size={11} />
                      {slot.time}
                    </div>
                  )}
                  <div className="itin-slot-place">{slot.place}</div>
                </div>
              </div>
            ))
          ) : (
            <div className="itin-slot-empty">Không có chi tiết giờ giấc</div>
          )}
        </div>
      )}
    </div>
  );
}

function ConstraintWarningBanner({ warning }) {
  if (!warning) return null;
  return (
    <div className="itin-constraint-warning" role="alert">
      <AlertTriangle size={16} className="itin-warning-icon" />
      <span>{warning.message}</span>
    </div>
  );
}

function LodgingSection({ lodgings }) {
  if (!lodgings || lodgings.length === 0) return null;
  return (
    <div className="itin-lodging">
      <div className="itin-lodging-header">
        <Hotel size={14} />
        <span>Gợi ý nghỉ đêm</span>
      </div>
      {lodgings.map((l, i) => (
        <div key={i} className="itin-lodging-item">
          <Moon size={12} className="itin-lodging-moon" />
          <span>{l}</span>
        </div>
      ))}
    </div>
  );
}

function RouteDistanceBadge({ summary }) {
  if (!summary) return null;
  return (
    <div className="itin-route-badge">
      <Car size={13} className="itin-route-icon" />
      <span>Tổng tuyến đường: <strong>{summary}</strong></span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

/**
 * ItineraryCard renders a parsed multi-day itinerary as a beautiful timeline UI.
 *
 * Props:
 *   text              {string}       — raw markdown itinerary text from AI
 *   constraintWarning {object|null}  — {message, coastal, sunset, island} or null
 *   routeSummary      {string}       — "X km • Y phút" from Mapbox, or empty
 */
const ItineraryCard = ({ text, constraintWarning, routeSummary }) => {
  const days = parseItineraryDays(text);
  const lodgings = parseLodging(text);

  // If we can't parse any days, fall back to null (caller will use ReactMarkdown)
  if (days.length === 0) return null;

  return (
    <div className="itin-card">
      <ConstraintWarningBanner warning={constraintWarning} />

      <div className="itin-days">
        {days.map((day, i) => (
          <DayCard key={i} day={day} index={i} defaultOpen={i === 0} />
        ))}
      </div>

      <LodgingSection lodgings={lodgings} />
      <RouteDistanceBadge summary={routeSummary} />
    </div>
  );
};

export default ItineraryCard;
