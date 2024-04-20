import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handlePredict = () => {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
      method: 'POST',
      body: formData,
    })
      .then(response => response.blob())
      .then(blob => {
        // Handle the predicted MIDI file
        // For example, you can create a download link for the user
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'predicted.mid';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
      })
      .catch(error => {
        console.error('Error predicting:', error);
      });
  };

  return (
    <div className="App">
      <h1>AI powered melody completion shyshtum</h1>
      <div className="card">
        <input type="file" accept=".mid,.midi" onChange={handleFileChange} />
        <button onClick={handlePredict}>Predict</button>
        <p>
          Upload a .midi file and our melody completion AI will complete the music for you
        </p>
      </div>
      <p className="read-the-docs">
        Team Chingari Pvt Ltd.
      </p>
    </div>
  );
}

export default App;
