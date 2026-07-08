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
    positions.push({ index: match.index, length: match[0].length, num: parseInt(match[1], 10), header: match[0].trim() });
  }

  if (positions.length === 0) return [];

  positions.forEach((pos, i) => {
    const start = pos.index + pos.length;
    const end = i + 1 < positions.length ? positions[i + 1].index : text.length;
    const chunk = text.slice(start, end).trim();

    // Clean up chunk: remove footer sections so they aren't parsed as slots
    const footerPattern = /(?:🏨|💰|💡|⚠️|Gợi\s+ý\s+nghỉ\s+đêm|Ước\s+tính\s+chi\s+phí|Lưu\s+ý\s+thực\s+tế|Ràng\s+buộc\s+đã\s+tuân\s+thủ)/i;
    const footerMatch = chunk.match(footerPattern);
    const scheduleChunk = footerMatch ? chunk.slice(0, footerMatch.index).trim() : chunk;

    // Parse time slots: support both same-line and next-line place names
    const slots = [];
    const lines = scheduleChunk.split("\n");
    for (let j = 0; j < lines.length; j++) {
      const line = lines[j].trim();
      const timeMatch = line.match(/^[-*\s]*(\d{1,2}:\d{2}(?:\s*[-–—]\s*\d{1,2}:\d{2})?)[-*\s]*:?\s*(.*)$/);
      if (timeMatch) {
        const time = timeMatch[1].trim();
        let place = timeMatch[2].replace(/^\*+|\*+$/g, "").replace(/\*\*/g, "").trim();
        if (!place && j + 1 < lines.length) {
          const nextLine = lines[j + 1].trim();
          if (!nextLine.match(/^[-*\s]*\d{1,2}:\d{2}/)) {
            place = nextLine.replace(/^\*+|\*+$/g, "").replace(/\*\*/g, "").trim();
            j++; // Consume next line
          }
        }

        // Collect description lines after the place line
        let descriptionLines = [];
        let k = j + 1;
        while (k < lines.length) {
          const nextLine = lines[k].trim();
          if (nextLine.match(/^[-*\s]*\d{1,2}:\d{2}/)) {
            break;
          }
          if (nextLine.match(/^(?:Ngày|Day|NGÀY)\s*\d+/i)) {
            break;
          }
          if (footerPattern.test(nextLine)) {
            break;
          }
          if (nextLine) {
            descriptionLines.push(nextLine);
          }
          k++;
        }
        j = k - 1;

        const description = descriptionLines.join("\n").replace(/\*\*/g, "").trim();

        if (place) {
          slots.push({ time, place, description });
        }
      }
    }

    // If no time-based slots, try bullet points as places
    if (slots.length === 0) {
      const bulletPattern = /^[\*\-]\s+(.+)/gm;
      let bMatch;
      while ((bMatch = bulletPattern.exec(scheduleChunk)) !== null) {
        const line = bMatch[1].replace(/^\*+|\*+$/g, "").replace(/\*\*/g, "").trim();
        if (line) slots.push({ time: "", place: line });
      }
    }

    // Fallback: If still no slots, split by paragraph blocks and infer session (Morning/Noon/Afternoon/Evening)
    if (slots.length === 0) {
      const sessionHeaderRegex = /(?:^|\n)(Sáng|Trưa|Chiều|Tối|Nghỉ\s+trưa)\s*(?:\n|$)/gi;
      let matches = [];
      let match;
      sessionHeaderRegex.lastIndex = 0;
      while ((match = sessionHeaderRegex.exec(scheduleChunk)) !== null) {
        matches.push({ label: match[1].trim(), index: match.index, length: match[0].length });
      }

      if (matches.length > 0) {
        for (let idx = 0; idx < matches.length; idx++) {
          const start = matches[idx].index + matches[idx].length;
          const end = idx + 1 < matches.length ? matches[idx + 1].index : scheduleChunk.length;
          const block = scheduleChunk.slice(start, end).trim();
          if (!block) continue;
          
          const blockLines = block.split("\n").map(l => l.trim()).filter(Boolean);
          if (blockLines.length > 0) {
            let place = blockLines[0].replace(/^[-*\s]+/, "").replace(/\*\*/g, "").trim();
            let description = blockLines.slice(1).join("\n").replace(/\*\*/g, "").trim();
            slots.push({
              time: matches[idx].label,
              place,
              description
            });
          }
        }
      } else {
        const paragraphs = scheduleChunk.split(/\n\s*\n+/).map(p => p.trim()).filter(p => p.length > 15);
        const finalParagraphs = paragraphs.length <= 1 ? scheduleChunk.split("\n").map(p => p.trim()).filter(Boolean) : paragraphs;
        
        let currentSession = "Sáng";
        for (const p of finalParagraphs) {
          const textClean = p.trim();
          if (!textClean) continue;
          if (textClean.length < 15 && !/(?:sáng|trưa|chiều|tối)/i.test(textClean)) continue;

          let timeLabel = "";
          const lowerP = textClean.toLowerCase();
          if (lowerP.includes("sáng") || lowerP.includes("mở đầu") || lowerP.includes("khởi đầu") || lowerP.includes("bắt đầu")) {
            timeLabel = "Sáng";
            currentSession = "Sáng";
          } else if (lowerP.includes("trưa")) {
            if (lowerP.includes("nghỉ")) {
              timeLabel = "Nghỉ trưa";
              currentSession = "Nghỉ trưa";
            } else {
              timeLabel = "Trưa";
              currentSession = "Trưa";
            }
          } else if (lowerP.includes("chiều")) {
            timeLabel = "Chiều";
            currentSession = "Chiều";
          } else if (lowerP.includes("tối")) {
            timeLabel = "Tối";
            currentSession = "Tối";
          }

          if (!timeLabel) {
            if (currentSession === "Trưa" || currentSession === "Nghỉ trưa") {
              currentSession = "Chiều";
            }
            timeLabel = currentSession;
          }

          let cleanText = textClean.replace(/^[-*\s]+/, "");
          if (timeLabel) {
            cleanText = cleanText.replace(new RegExp(`^(?:buổi\\s+)?${timeLabel}[:\\s,\\-]*`, "i"), "");
          }

          let place = "";
          let description = "";
          const boldMatch = cleanText.match(/\*\*(.*?)\*\*/);
          
          if (boldMatch) {
            place = boldMatch[1].replace(/\*\*/g, "").trim();
            description = cleanText.replace(/\*\*/g, "").trim();
          } else {
            const stripped = cleanText.replace(/\*\*/g, "").trim();
            const firstDot = stripped.indexOf(".");
            if (firstDot > 10 && firstDot < 90) {
              place = stripped.slice(0, firstDot).trim();
              description = stripped.slice(firstDot + 1).trim();
            } else {
              const words = stripped.split(/\s+/);
              if (words.length > 8) {
                place = words.slice(0, 8).join(" ") + "...";
                description = stripped;
              } else {
                place = stripped;
                description = "";
              }
            }
          }

          if (place && place.length > 3) {
            slots.push({
              time: timeLabel,
              place: place.trim(),
              description: description.trim()
            });
          }
        }
      }
    }

    // Normalize slot sessions sequentially to guarantee a clean timeline structure for fallback slots
    const sessionLabels = ["Sáng", "Trưa", "Nghỉ trưa", "Chiều", "Tối"];
    if (slots.length > 0 && slots.every(s => !s.time || s.time === "Sáng")) {
      if (slots.length === 4) {
        slots[0].time = "Sáng";
        slots[1].time = "Trưa";
        slots[2].time = "Nghỉ trưa";
        slots[3].time = "Chiều";
      } else if (slots.length === 3) {
        slots[0].time = "Sáng";
        slots[1].time = "Trưa";
        slots[2].time = "Chiều";
      } else if (slots.length === 2) {
        slots[0].time = "Sáng";
        slots[1].time = "Chiều";
      } else {
        for (let idx = 0; idx < slots.length; idx++) {
          if (idx < sessionLabels.length) {
            slots[idx].time = sessionLabels[idx];
          }
        }
      }
    }

    // Extract region label: "📍 Khu vực: ..." or "**Khu vực:**"
    const regionMatch = scheduleChunk.match(/(?:📍\s*)?(?:Khu\s+vực|Khu vuc)[:\s]+([^\n]+)/i);
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
                  {slot.description && (
                    <div className="itin-slot-desc">{slot.description}</div>
                  )}
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
