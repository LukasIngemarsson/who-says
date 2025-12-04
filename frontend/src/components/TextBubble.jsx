import { Box, Paper, Typography, Avatar } from "@mui/material";

// Helper to format seconds into MM:SS
const formatTime = (seconds) => {
  const min = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${min}:${sec < 10 ? "0" : ""}${sec}`;
};

// Generate a consistent color based on speaker ID
const getBubbleColor = (id) => {
  const colors = ["#e3f2fd", "#f3e5f5", "#e8f5e9", "#fff3e0"];
  return colors[id % colors.length];
};

const TextBubble = ({ speakerName, text, startTime, endTime, clusterId }) => {
  const bgColor = getBubbleColor(clusterId);

  return (
    <Box sx={{ display: "flex", marginBottom: 2, alignItems: "flex-start" }}>
      {/* Avatar Circle */}
      <Avatar
        sx={{
          bgcolor: "primary.main",
          marginRight: 2,
          marginTop: 1,
          width: 40,
          height: 40,
          fontSize: "1rem",
        }}
      >
        {speakerName?.charAt(0).toUpperCase() || "?"}
      </Avatar>

      {/* Message Bubble */}
      <Paper
        elevation={1}
        sx={{
          padding: 2,
          backgroundColor: bgColor,
          borderRadius: 3,
          maxWidth: "80%",
          position: "relative",
        }}
      >
        {/* Header: Name and Time */}
        <Box
          display="flex"
          justifyContent="space-between"
          alignItems="center"
          marginBottom={0.5}
        >
          <Typography
            variant="subtitle2"
            fontWeight="bold"
            color="text.primary"
            sx={{ mr: 2 }}
          >
            {speakerName}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {formatTime(startTime)} - {formatTime(endTime)}
          </Typography>
        </Box>

        {/* The spoken text */}
        <Typography
          variant="body1"
          color="text.primary"
          sx={{ lineHeight: 1.5 }}
        >
          {text}
        </Typography>
      </Paper>
    </Box>
  );
};

export default TextBubble;
