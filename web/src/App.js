import React, { useEffect, useRef, useState } from 'react';
import './App.css';

import WasmMosaic from './WasmMosaic';

function App() {

    const canvasRef = useRef(null);
    const [text, setText] = useState('');
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
            const mosaic = new WasmMosaic(instance);
            setWasmMosaic(mosaic);
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

        let animationFrameId;

        const animate = () => {
            if (!isPlaying || !wasmMosaic || wasmMosaic.empty()) return;

            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');

            const { width, height } = wasmMosaic.getSize();
            canvas.width = width;
            canvas.height = height;

            let step_was_valid;
            if (!computationComplete) {
                step_was_valid = wasmMosaic.stepK(k);
                if (!step_was_valid) {
                    setComputationComplete(true);
                }
                
            }
            
            const start = wasmMosaic.getRenderPointer();
            wasmMosaic.renderImageRange(start, k);

            const dataArray = wasmMosaic.getRawData();
            const imageData = new ImageData(dataArray, width, height);
            ctx.putImageData(imageData, 0, 0);

            
            
            animationFrameId = requestAnimationFrame(animate);
        };

        if (isPlaying) { 
            animationFrameId = requestAnimationFrame(animate);
        }

        return () => cancelAnimationFrame(animationFrameId);


    }, [isPlaying, wasmMosaic, computationComplete]);




    const handleUpload = async (event) => { 
        const file = event.target.files[0];
        if (!file || !wasmMosaic) return;

        setIsPlaying(false);

        if (!wasmMosaic.empty()) { 
            wasmMosaic.destroy();
        }



        const arrayBuffer = await file.arrayBuffer();
        const byteArray = new Uint8Array(arrayBuffer);
        const size = byteArray.length;
        wasmMosaic.loadMosaicFromBytes(byteArray, size);

        const { width, height } = wasmMosaic.getSize();
        const size_string = "Mosaic: " + width + " x " + height;
        setText(size_string)

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
