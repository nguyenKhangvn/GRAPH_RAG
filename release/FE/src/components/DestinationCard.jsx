import React from "react";
import { Star, CheckCircle2 } from "lucide-react";

const DestinationCard = ({ place, isVisible, x, y, onClose }) => {
  return (
    <div
      className={`destination-card ${isVisible ? "visible" : ""}`}
      style={{ left: x, top: y }}
    >
      <img src={place.image} alt={place.name} className="card-image" />
      <div className="card-content">
        <div className="card-header">
          <h3 className="card-title">{place.name}</h3>
          {onClose && (
            <button type="button" className="card-btn" onClick={onClose}>
              Đóng
            </button>
          )}
          {place.verified && (
            <span className="badge-verified">
              <CheckCircle2 size={12} />
              Verified
            </span>
          )}
        </div>
        <div className="card-rating">
          <Star className="star-icon" size={14} />
          <span>
            {place.rating} ({place.reviews} reviews)
          </span>
        </div>
        <p className="card-desc">{place.description}</p>
        <button className="card-btn">View Details</button>
      </div>
    </div>
  );
};

export default DestinationCard;
