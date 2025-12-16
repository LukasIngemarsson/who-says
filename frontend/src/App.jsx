import { useState } from 'react';
import './App.css'; // You can keep standard vite css or add your own

function App() {
  const [file, setFile] = useState(null);
  const [numSpeakers, setNumSpeakers] = useState(2);
  const [status, setStatus] = useState('');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setStatus('Please select a file first.');
      return;
    }

    setIsLoading(true);
    setStatus('Uploading and processing...');
    setResults(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_speakers', numSpeakers);

    try {
      // Notice we just call /process. The Vite proxy handles the localhost:5000 part.
      const response = await fetch('/process', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'An unknown error occurred');
      }

      setStatus('Processing complete!');
      setResults(data);
    } catch (error) {
      console.error('Error:', error);
      setStatus(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <h2>WhoSays React Interface</h2>
      
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '15px', maxWidth: '400px' }}>
        <div>
          <label>Audio File:</label>
          <input type="file" accept="audio/*" onChange={handleFileChange} required />
        </div>
        
        <div>
          <label>Number of Speakers:</label>
          <input 
            type="number" 
            value={numSpeakers} 
            onChange={(e) => setNumSpeakers(e.target.value)}
            min="1"
            style={{ marginLeft: '10px', width: '50px' }}
          />
        </div>

        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Processing...' : 'Process Audio'}
        </button>
      </form>

      <div style={{ marginTop: '20px', fontWeight: 'bold' }}>{status}</div>

      {results && (
        <pre style={{ background: '#222', color: '#0f0', padding: '15px', borderRadius: '5px', overflowX: 'auto' }}>
          {JSON.stringify(results, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default App;