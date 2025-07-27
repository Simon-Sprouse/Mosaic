import React, { useEffect, useRef, useState } from 'react';
import './App.css';

import WasmImage from './WasmImage';

function App() {

  const [cppText, setCppText] = useState('');
  const [wasmImage, setWasmImage] = useState(null);

  // LOAD WASM + SAVE MODULE IN CLASS
  useEffect(() => {
    const script = document.createElement('script');
    script.src = process.env.PUBLIC_URL + '/wasm/image_module.js';

    script.onload = () => {
      window.Module().then((instance) => {
        const img = new WasmImage(instance);
        setWasmImage(img);
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



  const handleClick = () => {
    if (!wasmImage) {
      setCppText('WASM not loaded yet');
      return;
    }

    const message = wasmImage.getSizeStr();
    setCppText(message);
  };

  return (
    <div className="App">
      <header className="App-header">
        <p>{cppText || "Click the button to run C++ code"}</p>
        <button onClick={handleClick}>Call C++</button>
      </header>
    </div>
  );
}

export default App;
