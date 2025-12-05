export const formatTime = (time) => {
  if (!time && time !== 0) return "--:--";
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
};

export const SPEAKER_COLORS = {
  0: "#3b82f6", // Blue
  1: "#10b981", // Emerald
  2: "#8b5cf6", // Violet
  default: "#64748b", // Slate
};


// TODO: test this!!!!
export const generateSpeakerColors = (numSpeakers) => {
  const colors = [];
  const baseColors = Object.values(SPEAKER_COLORS).filter(
    (color, index) => index < Object.keys(SPEAKER_COLORS).length - 1
  );
  for (let i = 0; i < numSpeakers; i++) {
    colors.push(baseColors[i % baseColors.length]);
  }
  return colors;
}