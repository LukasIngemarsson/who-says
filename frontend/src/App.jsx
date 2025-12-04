import "./App.css";
import UploadAudioFile from "./components/UploadAudioFile.jsx";
import User from "./components/User.jsx";
import TextBubble from "./components/TextBubble.jsx";
import { useState } from "react";
import { Box, Stack, Container } from "@mui/material";

function App() {
  const [userNames, setUserNames] = useState();
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <h2>WhoSays?</h2>

        <UploadAudioFile
          userNames={userNames}
          setUserNames={setUserNames}
          setResults={setResults}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
        />

        {/* List of identified users at the top */}
        <Box display="flex" gap={2} my={2} flexWrap="wrap">
          {userNames?.map((name, idx) => (
            <User name={name} key={idx} />
          ))}
        </Box>

        {/* The Conversation Feed */}
        {results && results.segments && (
          <Box sx={{ mt: 4, display: "flex", flexDirection: "column" }}>
            {results.segments.map((segment, index) => {
              // Try to find the name from the userNames array using cluster_id
              // If not found, use the raw speaker label (e.g., "SPEAKER_0")
              const displayName =
                userNames?.[segment.cluster_id] || segment.speaker;

              return (
                <TextBubble
                  key={index}
                  speakerName={displayName}
                  text={segment.text}
                  startTime={segment.start}
                  endTime={segment.end}
                  clusterId={segment.cluster_id}
                />
              );
            })}
          </Box>
        )}
      </Box>
    </Container>
  );
}

export default App;
