import { useEffect, useRef, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// Cache GPS location as "lat,lng" string. Empty if unavailable.
let cachedGeoLocation = "";
let geoLocationReady = false;

function requestGeoLocation() {
  if (!navigator.geolocation) {
    console.warn("[Geo] navigator.geolocation not available");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      cachedGeoLocation = `${pos.coords.latitude},${pos.coords.longitude}`;
      geoLocationReady = true;
      console.log("[Geo] Location acquired:", cachedGeoLocation);
    },
    (err) => {
      console.warn("[Geo] Location error:", err.message);
    },
    { enableHighAccuracy: false, timeout: 10000 }
  );
}

export const useChatStreaming = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      content:
        "Xin chào! Tôi là trợ lý du lịch ảo của Gia Lai. Bạn muốn tìm hiểu về địa điểm nào hôm nay? Hãy thử hỏi về các thác nước hoặc chùa chiền nhé!",
      isStreaming: false,
    },
  ]);
  const [mapLocations, setMapLocations] = useState([]);
  const [mapIntent, setMapIntent] = useState(null);
  const [mapGraph, setMapGraph] = useState({ nodes: [], links: [] });
  const [mapDistance, setMapDistance] = useState(null);
  const [mapSafety, setMapSafety] = useState(null);
  const [mapConstraintWarning, setMapConstraintWarning] = useState(null);
  const [mapDailyPlan, setMapDailyPlan] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [detectedLocation, setDetectedLocation] = useState("");

  const lastQueryRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    requestGeoLocation();
  }, []);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const addMessage = (message) => {
    setMessages((prev) => [...prev, message]);
  };

  const updateLastMessage = (updater) => {
    setMessages((prev) => {
      if (!prev.length) return prev;
      const updated = [...prev];
      const last = updated[updated.length - 1];
      updated[updated.length - 1] =
        typeof updater === "function" ? updater(last) : { ...last, ...updater };
      return updated;
    });
  };

  const handleSend = async (userQuery) => {
    if (!userQuery.trim() || isTyping) return;

    lastQueryRef.current = userQuery;

    const newUserMsg = { id: Date.now(), role: "user", content: userQuery };
    const initialAiMsg = {
      id: Date.now() + 1,
      role: "assistant",
      content: "",
      isError: false,
      isStreaming: true,
    };

    const historyPayload = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    addMessage(newUserMsg);
    addMessage(initialAiMsg);
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          query: userQuery,
          chat_history: historyPayload,
          current_location: cachedGeoLocation,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Server trả về lỗi ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();

        if (done) {
          updateLastMessage({ isStreaming: false });
          setIsTyping(false);
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const eventBlocks = buffer.split("\n\n");
        buffer = eventBlocks.pop() || "";

        for (const block of eventBlocks) {
          if (!block.trim()) continue;

          let eventName = "message";
          let dataStr = "";

          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) {
              eventName = line.replace("event:", "").trim();
            } else if (line.startsWith("data:")) {
              dataStr = line.replace("data:", "").trim();
            }
          }

          if (!dataStr) continue;

          if (eventName === "done" && dataStr === "[DONE]") {
            await reader.cancel();
            updateLastMessage({ isStreaming: false });
            setIsTyping(false);
            break;
          }

          if (eventName === "message") {
            try {
              const { chunk } = JSON.parse(dataStr);
              if (!chunk) continue;

              updateLastMessage((last) => ({
                ...last,
                content: `${last.content}${chunk}`,
                isStreaming: true,
                isError: false,
              }));
              scrollToBottom();
            } catch (err) {
              console.error("Parse error (message):", err);
            }
            continue;
          }

          if (eventName === "metadata") {
            try {
              const data = JSON.parse(dataStr);
              if (data.intent) setMapIntent(data.intent);
              if (data.detected_location) {
                setDetectedLocation(data.detected_location);
              }
              if (Array.isArray(data.locations)) {
                setMapLocations(data.locations);
              }
              if (
                data.graph &&
                Array.isArray(data.graph.nodes) &&
                Array.isArray(data.graph.links)
              ) {
                setMapGraph(data.graph);
              }
              if (data.distance && typeof data.distance === "object") {
                setMapDistance(data.distance);
              } else {
                setMapDistance(null);
              }
              if (
                data.tour_plan_safety &&
                typeof data.tour_plan_safety === "object"
              ) {
                setMapSafety(data.tour_plan_safety);
              } else {
                setMapSafety(null);
              }
              // Constraint warning: shown when coastal/sunset/island not satisfied
              if (data.constraint_warning && typeof data.constraint_warning === "object") {
                setMapConstraintWarning(data.constraint_warning);
                // Also attach to the AI message so ItineraryCard can render it
                updateLastMessage((last) => ({
                  ...last,
                  constraintWarning: data.constraint_warning,
                }));
              } else {
                setMapConstraintWarning(null);
              }
              // Daily cluster plan: used to build day badges in MapInterface
              if (Array.isArray(data.daily_cluster_plan)) {
                setMapDailyPlan(data.daily_cluster_plan);
              } else {
                setMapDailyPlan([]);
              }
            } catch (err) {
              console.error("Parse error (metadata):", err);
            }
            continue;
          }

          if (eventName === "error") {
            try {
              const { error } = JSON.parse(dataStr);
              updateLastMessage({
                content: `⚠️ ${error || "Lỗi xử lý yêu cầu"}`,
                isError: true,
                isStreaming: false,
              });
            } catch (err) {
              console.error("Parse error (error event):", err);
            }
          }
        }
      }
    } catch (error) {
      console.error("Network/API Error:", error);
      updateLastMessage({
        content: "Không thể kết nối tới server. Vui lòng thử lại.",
        isError: true,
        isStreaming: false,
      });
      setIsTyping(false);
    }
  };

  const handleRetry = () => {
    if (!lastQueryRef.current || isTyping) return;

    setMessages((prev) => {
      const updated = [...prev];
      if (updated[updated.length - 1]?.isError) {
        updated.pop();
      }
      return updated;
    });

    handleSend(lastQueryRef.current);
  };

  return {
    messages,
    mapLocations,
    mapIntent,
    mapGraph,
    mapDistance,
    mapSafety,
    mapConstraintWarning,
    mapDailyPlan,
    isTyping,
    handleSend,
    handleRetry,
    chatEndRef,
    detectedLocation,
  };
};
