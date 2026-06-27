import React, { useState } from "react";
import { Send, Map as MapIcon, ChevronRight, RotateCcw } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ItineraryCard from "./ItineraryCard";

const SUGGESTIONS = [
  {
    label: "📍 Địa điểm nổi tiếng?",
    query: "Địa điểm du lịch nổi tiếng ở Gia Lai?",
  },
  {
    label: "🍜 Đặc sản phải thử?",
    query: "Đặc sản ẩm thực Gia Lai có gì ngon?",
  },
  {
    label: "🏨 Khách sạn ở Pleiku?",
    query: "Khách sạn tốt ở Pleiku, Gia Lai?",
  },
  {
    label: "🌊 Lịch trình có biển?",
    query: "Lịch trình 2 ngày ở Bình Định yêu cầu có biển?",
  },
  {
    label: "🎭 Lễ hội văn hóa?",
    query: "Lễ hội và sự kiện văn hóa ở Gia Lai?",
  },
  {
    label: "🗺️ Tour 2 ngày?",
    query: "Gợi ý lịch trình tour du lịch Gia Lai 2 ngày?",
  },
];

const formatTime = (id) => {
  const d = new Date(id > 1e12 ? id : Date.now());
  return d.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
};

/**
 * Returns true if the message text looks like a multi-day itinerary.
 * We check for "Ngày 1" / "Ngày 2" header patterns (with or without markdown).
 */
const isItineraryMessage = (text) => {
  if (!text) return false;
  // Requires at least "Ngày 1" AND "Ngày 2" to be an itinerary
  const hasDay1 = /(?:^|\n).{0,10}?(?:Ngày|Day|NGÀY)\s*1\b/i.test(text);
  const hasDay2 = /(?:^|\n).{0,10}?(?:Ngày|Day|NGÀY)\s*2\b/i.test(text);
  return hasDay1 && hasDay2;
};

/**
 * Extract the non-itinerary portions of text (after all day sections):
 * sections like "Lưu ý", "Chi phí", "Gợi ý nghỉ đêm" that appear
 * after the last "Ngày X" block — we keep those for ReactMarkdown.
 */
const extractNonItineraryText = (text) => {
  if (!text) return "";
  // Find the last occurrence of a day header
  const dayPattern = /(?:^|\n).{0,10}?(?:Ngày|Day|NGÀY)\s*\d+[^\n]*/gi;
  let lastIndex = 0;
  let m;
  while ((m = dayPattern.exec(text)) !== null) {
    lastIndex = m.index + m[0].length;
  }
  // Everything after the last day header's first line
  const rest = text.slice(lastIndex).trim();
  // Remove time-slot lines from rest (they are rendered in ItineraryCard)
  return rest
    .split("\n")
    .filter((line) => !/^\s*[\*\-]?\s*\d{1,2}:\d{2}/.test(line))
    .join("\n")
    .trim();
};


const ChatInterface = ({
  messages,
  isTyping,
  onSend,
  onRetry,
  chatEndRef,
  detectedLocation,
  routeSummary,
}) => {
  const [input, setInput] = useState("");
  const showSuggestions = messages.length <= 1 && !isTyping;

  const handleFormSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isTyping) return;
    onSend(input);
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleFormSubmit(e);
    }
  };

  // Find the index of the last assistant itinerary message
  const lastItinIndex = messages
    .map((m, i) => ({ role: m.role, isError: m.isError, isItin: isItineraryMessage(m.content), i }))
    .reverse()
    .find((m) => m.role === "assistant" && !m.isError && m.isItin)?.i;

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="chat-avatar">
          <MapIcon size={18} color="white" />
        </div>
        <div className="chat-header-info">
          <span className="chat-header-title">Gia Lai Travel AI</span>
          <span className="chat-header-sub">
            {detectedLocation
              ? `📍 ${detectedLocation}`
              : "Trợ lý du lịch thông minh"}
          </span>
        </div>
        <div className="chat-status-dot" title="Đang hoạt động" />
      </div>

      <div className="chat-messages scrollable">
        {messages.map((msg, index) => {
          const isActivelyStreaming =
            index === messages.length - 1 &&
            msg.role === "assistant" &&
            msg.isStreaming;

          return (
            <div
              key={msg.id}
              className={`message-wrapper ${msg.role === "user" ? "message-wrapper-user" : "message-wrapper-ai"}`}
            >
              {msg.role === "assistant" && (
                <div className="message-avatar-small">
                  <MapIcon size={12} color="white" />
                </div>
              )}
              <div className="message-col">
                <div
                  className={`message-bubble ${
                    msg.role === "user"
                      ? "message-user"
                      : msg.isError
                        ? "message-ai message-error"
                        : "message-ai"
                  }`}
                >
                  <div className="markdown-body">
                    {/* Try to render as ItineraryCard if message looks like itinerary */}
                    {msg.role === "assistant" && !msg.isError && isItineraryMessage(msg.content) ? (
                      <>
                        <ItineraryCard
                          text={msg.content}
                          constraintWarning={msg.constraintWarning || null}
                          routeSummary={index === lastItinIndex ? routeSummary : null}
                        />
                        {/* Show non-itinerary parts (summary, notes) as markdown */}
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{extractNonItineraryText(msg.content)}</ReactMarkdown>
                      </>
                    ) : (
                      <>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                        {isActivelyStreaming && (
                          <span className="blinking-cursor"></span>
                        )}
                      </>
                    )}
                  </div>
                </div>
                {msg.isError && (
                  <button className="retry-btn" onClick={onRetry}>
                    <RotateCcw size={12} /> Thử lại
                  </button>
                )}
                <div
                  className={`message-time ${msg.role === "user" ? "message-time-right" : ""}`}
                >
                  {formatTime(msg.id)}
                </div>
              </div>
            </div>
          );
        })}
        <div ref={chatEndRef} />
      </div>

      {showSuggestions && (
        <div className="suggestions-wrapper">
          <p className="suggestions-label">Gợi ý câu hỏi</p>
          <div className="suggestions-grid">
            {SUGGESTIONS.map((s, i) => (
              <button
                key={i}
                className="suggestion-chip"
                onClick={() => {
                  if (!isTyping) onSend(s.query);
                }}
              >
                <span>{s.label}</span>
                <ChevronRight size={12} className="chip-arrow" />
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="chat-input-wrapper">
        <form
          onSubmit={handleFormSubmit}
          className="chat-input-bar glass-input"
        >
          <input
            type="text"
            className="chat-input"
            placeholder="Hỏi về địa điểm, ẩm thực, lưu trú..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isTyping}
            autoComplete="off"
          />
          <button
            type="submit"
            className="chat-send-btn"
            disabled={isTyping || !input.trim()}
            title="Gửi (Enter)"
          >
            <Send size={16} />
          </button>
        </form>
        <p className="input-hint">Nhấn Enter để gửi</p>
      </div>
    </div>
  );
};

export default ChatInterface;
