import React, { useEffect, useRef, useState } from 'react';
import './App.css';

function App() {
  const canvasRef = useRef(null);
  const [cppText, setCppText] = useState('');

  useEffect(() => {
    const script = document.createElement('script');
    script.src = process.env.PUBLIC_URL + '/wasm/image_module.js';

    script.onload = () => {
      // Now Module is a function, not a global object
      window.Module().then((instance) => {
        const ptr = instance._helloWorld();
        const message = instance.UTF8ToString(ptr);
        setCppText(message);
      }).catch((err) => {
        console.error('Failed to instantiate WASM module:', err);
      });
    };

    script.onerror = () => {
      console.error('Failed to load image_module.js');
    };

    document.body.appendChild(script);
    return () => {
      document.body.removeChild(script);
    };
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <canvas ref={canvasRef} />
        <p>{cppText || "Loading..."}</p>
      </header>
    </div>
  );
}

export default App;
