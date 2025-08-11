import React, { useEffect, useRef, useState, useCallback } from 'react';
import './App.css';

import WasmMosaic from './WasmMosaic';

function App() {

    const canvasRef = useRef(null);
    const animationFrameRef = useRef(null);

    const [text, setText] = useState('');
    const [wasmModule, setWasmModule] = useState(null);
    const [wasmMosaic, setWasmMosaic] = useState(null);
    const [animationMode, setAnimationMode] = useState("paused");
    const [computationComplete, setComputationComplete] = useState(false);

    const k = 100;
    const multi_step = 100;



    // Function to progress wasm computation (if needed) and handle forward animation
    const stepK = (k) => { 

        if (!wasmMosaic || wasmMosaic.empty()) return;

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


    }

    // Function to handle backward animation
    const reverseStepK = (k) => { 
        if (!wasmMosaic || wasmMosaic.empty()) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        const { width, height } = wasmMosaic.getSize();
        canvas.width = width;
        canvas.height = height;

        const current = wasmMosaic.getRenderPointer();
        
        wasmMosaic.resetCanvas();
        wasmMosaic.resetRenderPointer();
        wasmMosaic.renderImageRange(0, Math.max(0, current - k));

        const dataArray = wasmMosaic.getRawData();
        const imageData = new ImageData(dataArray, width, height);
        ctx.putImageData(imageData, 0, 0);
    }



    // LOAD WASM + SAVE MODULE IN STATE
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


    // Request Animation frame if play/reverse mode
    useEffect(() => {

        const animate = () => {

            if (animationMode === "paused") { 
                return;
            }
            else if (animationMode === "play") { 
                stepK(k);
            }
            else if (animationMode === "reverse") { 
                reverseStepK(k);
            }
            
            animationFrameRef.current = requestAnimationFrame(animate);
        };

        if (animationMode !== "paused") {
            animationFrameRef.current = requestAnimationFrame(animate);
        }

        return () => cancelAnimationFrame(animationFrameRef.current);
    }, [animationMode, wasmMosaic, computationComplete]);

    const stopAnimation = () => { 

        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

        setAnimationMode("paused");
    }




    const handleUpload = async (event) => { 
        const file = event.target.files[0];
        if (!file || !wasmModule) return;

        setAnimationMode(false);
        cancelAnimationFrame(animationFrameRef.current);
        setAnimationMode(false);
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

        // TODO load black canvas to show what dimesions will be
        const canvas = canvasRef.current;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, width, height);

    }



    const handlePlay = () => { 
        if (animationMode === "play") { 
            setAnimationMode("paused");
        }
        else {
            setAnimationMode("play");
        }
    }

    const handlePause = () => { 
        setAnimationMode("paused");
    }

    const handleReverse = () => { 
        if (animationMode === "reverse") {
            setAnimationMode("paused");
        }
        else { 
            setAnimationMode("reverse");
        }
    }

    const handleReplay = () => { 
        if (!wasmMosaic || wasmMosaic.empty()) return;
        wasmMosaic.resetRenderPointer();
        wasmMosaic.resetCanvas();

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        setAnimationMode("play");
    }

    const handleReset = () => { 

        setAnimationMode("paused");

        if (!wasmMosaic || wasmMosaic.empty()) return;
        wasmMosaic.resetRenderPointer();
        wasmMosaic.resetCanvas();

        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
    }

    const stepForward = () => {
        if (animationMode !== "paused") stopAnimation();
        stepK(1);
    };

    const stepBack = () => {
        if (animationMode !== "paused") stopAnimation();
        reverseStepK(1);
    }

    const stepForwardMulti = () => { 
        if (animationMode !== "paused") stopAnimation();
        stepK(multi_step);
    }

    const stepBackMulti = () => { 
        if (animationMode !== "paused") stopAnimation();
        reverseStepK(multi_step);
    }




    return (
        <div className="App">
        <header className="App-header">
            <input type="file" accept="image/*" onChange={handleUpload} />
            <p>{text || "Click the button to run C++ code"}</p>
            <button onClick={handlePlay}>
                Play
            </button>
            <button onClick={handlePause}>
                Pause
            </button>
            <button onClick={handleReverse}>
                Reverse
            </button>
            <button onClick={handleReplay}>
                Replay Animation
            </button>
            <button onClick={handleReset}>
                Reset Animation
            </button>
            <button onClick={stepForward}>
                Step Forward
            </button>
            <button onClick={stepBack}>
                Step Back
            </button>
            <button onClick={stepForwardMulti}>
                Step Forward Multi
            </button>
            <button onClick={stepBackMulti}>
                Step Back Multi
            </button>
            <canvas ref={canvasRef} />
        </header>
        </div>
    );
}

export default App;
