import React, { useEffect, useRef, useState } from 'react';
import './App.css';

import WasmMosaic from './WasmMosaic';

function App() {

    const canvasRef = useRef(null);
    const animationFrameRef = useRef(null);

    const [text, setText] = useState('');
    const [wasmModule, setWasmModule] = useState(null);
    const [wasmMosaic, setWasmMosaic] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [computationComplete, setComputationComplete] = useState(false);

    const k = 100;



    // LOAD WASM + SAVE MODULE IN CLASS
    useEffect(() => {
        const script = document.createElement('script');
        script.src = process.env.PUBLIC_URL + '/wasm/image_module.js';

        script.onload = () => {
        window.Module().then((instance) => {
            setWasmModule(instance);
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


    // Request Animation frame if play button
    useEffect(() => {
        const animate = () => {
            if (!isPlaying || !wasmMosaic || wasmMosaic.empty()) return;

            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');

            const { width, height } = wasmMosaic.getSize();
            canvas.width = width;
            canvas.height = height;

            if (!computationComplete) {
                const stepValid = wasmMosaic.stepK(k);
                if (!stepValid) setComputationComplete(true);
            }

            const start = wasmMosaic.getRenderPointer();
            wasmMosaic.renderImageRange(start, k);

            const dataArray = wasmMosaic.getRawData();
            const imageData = new ImageData(dataArray, width, height);
            ctx.putImageData(imageData, 0, 0);

            animationFrameRef.current = requestAnimationFrame(animate);
        };

        if (isPlaying) {
            animationFrameRef.current = requestAnimationFrame(animate);
        }

        return () => cancelAnimationFrame(animationFrameRef.current);
    }, [isPlaying, wasmMosaic, computationComplete]);




    const handleUpload = async (event) => { 
        const file = event.target.files[0];
        if (!file || !wasmModule) return;

        setIsPlaying(false);
        cancelAnimationFrame(animationFrameRef.current);
        setIsPlaying(false);
        setComputationComplete(false);

        if (wasmMosaic && !wasmMosaic.empty()) { 
            wasmMosaic.destroy();
        }

        const mosaic = new WasmMosaic(wasmModule);
        



        const arrayBuffer = await file.arrayBuffer();
        const byteArray = new Uint8Array(arrayBuffer);
        const size = byteArray.length;
        mosaic.loadMosaicFromBytes(byteArray, size);

        const { width, height } = mosaic.getSize();
        const size_string = "Mosaic: " + width + " x " + height;
        setText(size_string)

        setWasmMosaic(mosaic);

    }

    function handleReset() { 
        if (!wasmMosaic || wasmMosaic.empty()) return;
        wasmMosaic.resetRenderPointer();
        wasmMosaic.resetCanvas();

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        setIsPlaying(true);
    }

    return (
        <div className="App">
        <header className="App-header">
            <input type="file" accept="image/*" onChange={handleUpload} />
            <p>{text || "Click the button to run C++ code"}</p>
            <button onClick={() => setIsPlaying(prev => !prev)}>
                {isPlaying ? "Pause" : "Play"}
            </button>
            <button onClick={handleReset}>
                Reset Animation
            </button>
            <canvas ref={canvasRef} />
        </header>
        </div>
    );
}

export default App;
