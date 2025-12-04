import { useState } from "react";

const UploadAudioFile = ({
  userNames,
  setUserNames,
  setResults,
  isLoading,
  setIsLoading,
}) => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setStatus("Please select a file first.");
      return;
    }

    if (!userNames || userNames.length === 0) {
      setStatus("Please enter at least one speaker name.");
      return;
    }

    setIsLoading(true);
    setStatus("Uploading and processing...");
    setResults(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_speakers", userNames?.length);

    try {
      // Notice we just call /process. The Vite proxy handles the localhost:5000 part.
      const response = await fetch("/process", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "An unknown error occurred");
      }

      setStatus("Processing complete!");
      setResults(data);
    } catch (error) {
      console.error("Error:", error);
      setStatus(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <form
        onSubmit={handleSubmit}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "15px",
          maxWidth: "400px",
        }}
      >
        <div>
          <label>Audio File:</label>
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            required
          />
        </div>

        <div>
          <label>Number of Speakers:</label>
          <p>{userNames?.length ?? 0}</p>
        </div>

        <div>
          <label>Speaker Names (comma separated):</label>
          <input
            type="text"
            value={userNames?.join(",")}
            onChange={(e) =>
              e.target.value != ""
                ? setUserNames(
                    e.target.value.split(",").map((name) => name.trim())
                  )
                : setUserNames([])
            }
            placeholder="e.g. Alice, Bob, Charlie"
            style={{ marginLeft: "10px", width: "250px" }}
          />
        </div>

        <button type="submit" disabled={isLoading}>
          {isLoading ? "Processing..." : "Process Audio"}
        </button>
      </form>

      <div style={{ marginTop: "20px", fontWeight: "bold" }}>{status}</div>
    </div>
  );
};

export default UploadAudioFile;
