import React, { useEffect, useRef, useState } from 'react';
import './App.css';

import WasmImage from './WasmImage';

function App() {

    const canvasRef = useRef(null);
    const [text, setText] = useState('');
    const [wasmImage, setWasmImage] = useState(null);



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
            setText('WASM not loaded yet');
            return;
        }

        if (wasmImage.empty()) { 
            setText("No Image uploaded");
            return;
        }

        
        const { width, height } = wasmImage.getSize();
        const canvas = canvasRef.current;
        canvas.width = width;
        canvas.height = height;
        const text_string = "Image: " + width + " x " + height;
        setText(text_string);

        const ctx = canvas.getContext('2d');
        const dataArray = wasmImage.getDataArray();
        const imageData = new ImageData(dataArray, width, height);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(imageData, 0, 0);


        
        console.log("canvas updated from c++");

    };


    const handleUpload = async (event) => { 
        const file = event.target.files[0];
        if (!file || !wasmImage) return;

        if (!wasmImage.empty()) { 
            wasmImage.destroy();
        }

        const arrayBuffer = await file.arrayBuffer();
        const byteArray = new Uint8Array(arrayBuffer);
        wasmImage.loadImageFromBytes(byteArray);

    }

    return (
        <div className="App">
        <header className="App-header">
            <input type="file" accept="image/*" onChange={handleUpload} />
            <p>{text || "Click the button to run C++ code"}</p>
            <button onClick={handleClick}>Call C++</button>
            <canvas ref={canvasRef} />
        </header>
        </div>
    );
}

export default App;
