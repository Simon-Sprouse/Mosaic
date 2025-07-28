import React, { useEffect, useRef, useState } from 'react';
import './App.css';

import WasmImage from './WasmImage';

function App() {

    const canvasRef = useRef(null);
    const [cppText, setCppText] = useState('');
    const [wasmImage, setWasmImage] = useState(null);

    const width = 800;
    const height = 400;

    // LOAD WASM + SAVE MODULE IN CLASS
    useEffect(() => {
        const script = document.createElement('script');
        script.src = process.env.PUBLIC_URL + '/wasm/image_module.js';

        script.onload = () => {
        window.Module().then((instance) => {
            const img = new WasmImage(instance);
            // img.constructImage(width, height);
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


        // very clumsy way to manage memory
        wasmImage.destroy();
        wasmImage.constructImage(width, height);

        const message = wasmImage.getSizeStr();
        setCppText(message);

        const canvas = canvasRef.current;
        canvas.width = wasmImage.width;
        canvas.height = wasmImage.height;
        const ctx = canvas.getContext('2d');
        const dataArray = wasmImage.getDataArray();
        const imageData = new ImageData(dataArray, width, height);
        ctx.putImageData(imageData, 0, 0);


        console.log("canvas updated from c++");

    };

    return (
        <div className="App">
        <header className="App-header">
            <p>{cppText || "Click the button to run C++ code"}</p>
            <button onClick={handleClick}>Call C++</button>
            <canvas ref={canvasRef} />
        </header>
        </div>
    );
}

export default App;
