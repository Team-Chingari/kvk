import { useState } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  // Function to handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0]; // Get the uploaded file
    // You can now work with the uploaded file, such as sending it to your backend for processing
    console.log('Uploaded file:', file);
  }

  return (
    <>
      <h1>AI powered melody completion shyshtum</h1>
      <div className="card">
        {/* Input field for file upload */}
        <input type="file" accept=".midi" onChange={handleFileUpload} />
        {/* Button for uploading */}
        {/* You can remove this button if you prefer to use the default file upload UI */}
        {/* <button onClick={() => {}}>
          Upload
        </button> */}
        <p>
          Upload a .midi file and our melody completion AI will complete the music for you
        </p>
      </div>
      <p className="read-the-docs">
        Team Chingari Pvt Ltd.
      </p>
    </>
  )
}

export default App
