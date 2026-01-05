export const formatTime = (time) => {
  if (!time && time !== 0) return "--:--";
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
};

export const getSpeakerColor = (index) => {
  // Robust palette for the first 12 speakers
  const palette = [
    '#3b82f6', // Blue-500
    '#10b981', // Emerald-500
    '#8b5cf6', // Violet-500
    '#f59e0b', // Amber-500
    '#ec4899', // Pink-500
    '#ef4444', // Red-500
    '#06b6d4', // Cyan-500
    '#84cc16', // Lime-500
    '#d946ef', // Fuchsia-500
    '#f97316', // Orange-500
    '#6366f1', // Indigo-500
    '#14b8a6', // Teal-500
  ];

  if (index < palette.length) {
    return palette[index];
  }

  // Algorithmically generate distinct colors for high speaker counts (Golden Angle approximation)
  // This ensures colors remain distinct even if numSpeakers > 12
  const hue = (index * 137.508) % 360; 
  return `hsl(${hue}, 70%, 50%)`;
};