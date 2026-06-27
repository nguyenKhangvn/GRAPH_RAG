import React, { Suspense, lazy, useState } from "react";
import ChatInterface from "./components/ChatInterface";
import { useChatStreaming } from "./hooks/useChatStreaming";
import "./components.css";

const MapInterface = lazy(() => import("./components/MapInterface"));

const isItineraryMessage = (text) => {
  if (!text) return false;
  const hasDay1 = /(?:^|\n).{0,10}?(?:Ngày|Day|NGÀY)\s*1\b/i.test(text);
  const hasDay2 = /(?:^|\n).{0,10}?(?:Ngày|Day|NGÀY)\s*2\b/i.test(text);
  return hasDay1 && hasDay2;
};

function App() {
  const {
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
  } = useChatStreaming();

  const [routeSummary, setRouteSummary] = useState("");

  const lastItineraryText = [...messages]
    .reverse()
    .find((m) => m.role === "assistant" && !m.isError && isItineraryMessage(m.content))?.content || "";

  return (
    <div className="app-container">
      <ChatInterface
        messages={messages}
        isTyping={isTyping}
        onSend={handleSend}
        onRetry={handleRetry}
        chatEndRef={chatEndRef}
        detectedLocation={detectedLocation}
        routeSummary={routeSummary}
      />
      <Suspense fallback={<div className="map-container" />}>
        <MapInterface
          mapLocations={mapLocations}
          mapIntent={mapIntent}
          mapGraph={mapGraph}
          mapDistance={mapDistance}
          mapSafety={mapSafety}
          mapConstraintWarning={mapConstraintWarning}
          mapDailyPlan={mapDailyPlan}
          routeSummary={routeSummary}
          setRouteSummary={setRouteSummary}
          lastItineraryText={lastItineraryText}
        />
      </Suspense>
    </div>
  );
}

export default App;
