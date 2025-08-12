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

    const default_params = {
        resize_factor: 2,
        blur_kernel_size: 3,
        blur_sigma: 1.4,
        canny_threshold_1: 50,
        canny_threshold_2: 100,
        max_segment_angle_deg: 100,
        min_segment_legnth: 20,
        segment_angle_window: 10,
        tile_size: 20,
        number_of_rings: 3,
        intiial_step_factor: 1.5,
        step_size_factor: 0.25,
        min_intersection_distance_factor: 0.25,
        max_frontiers: 4,
        flood_fill_neighbor_points: 4,
        flood_fill_distance_factor: 1.5,
        max_background_points: 50000,
    }
    const [params, setParams] = useState(default_params);
    const updateParam = (key, value) => {
        setParams(prev => ({
            ...prev,
            [key]: parseFloat(value)
        }));
    };



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

        const current = wasmMosaic.getRenderPointer();
        wasmMosaic.renderImageRange(current, k);

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

    const clearOnscreenCanvas = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    const stopAnimation = () => { 

        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

        setAnimationMode("paused");
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


    // Handle dynamic parameters change from user settings
    useEffect(() => {
        if (!wasmMosaic) return;

        // 2. Stop animation if running 
        stopAnimation();

        // 3. clear old data
        wasmMosaic.clearData();

        // 4. update mosaic with new params
        wasmMosaic.setParameters(params);

        // 5. reset canvas
        wasmMosaic.resizeOriginal(); // done so we can access mosaic.size() and resize canvas
        const { width, height } = wasmMosaic.getSize();
        const canvas = canvasRef.current;
        canvas.width = width;
        canvas.height = height;
        clearOnscreenCanvas();

        // 6. mark comuptation flag as incomplete
        setComputationComplete(false);

        // 7. restart animation
        setTimeout(() => {
            setAnimationMode("play");
        }, 0);


    }, [params]); // <- run this effect when params change


    




    const handleUpload = async (event) => { 
        const file = event.target.files[0];
        if (!file || !wasmModule) return;

        // stop animation and reset computation flag
        stopAnimation();
        setComputationComplete(false);

        // destroy old mosaic
        if (wasmMosaic && !wasmMosaic.empty()) { 
            wasmMosaic.destroy();
        }

        // create new mosaic with new image
        const mosaic = new WasmMosaic(wasmModule);
        mosaic.createMosaic(params);

        // load byte array into mosaic
        const arrayBuffer = await file.arrayBuffer();
        const byteArray = new Uint8Array(arrayBuffer);
        mosaic.loadMosaicFromBytes(byteArray, byteArray.length);
        

        // display mosaic metadata
        const { width, height } = mosaic.getSize();
        const size_string = "Mosaic: " + width + " x " + height;
        setText(size_string)

        // store mosaic into state (this is async)
        setWasmMosaic(mosaic);


        // load black canvas to show what dimesions will be // TODO show image eventually
        const canvas = canvasRef.current;
        canvas.width = width;
        canvas.height = height;
        clearOnscreenCanvas();

    }



    /*
    --------------------------------------
        vvv FUNCTIONS FOR BUTTONS/IO vvv
    --------------------------------------
    */


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
            



            <div
            style={{
                display: 'flex',
                height: '100vh',
                boxSizing: 'border-box',
                padding: '1rem',
                gap: '1rem',
            }}
            >
            <div
            className="params-container"
            style={{
                maxWidth: 400,
                margin: '1rem auto',
                padding: '1rem',
                border: '1px solid #ccc',
                borderRadius: 8,
                backgroundColor: '#000000',
                overflowY: 'auto',
                maxHeight: '70vh',
            }}
            >
                <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Resize Factor: {params.resize_factor}
                    <input
                    type="range"
                    min="0.1"
                    max="5"
                    step="0.1"
                    value={params.resize_factor}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, resize_factor: parseFloat(e.target.value) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label>

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Blur Kernel Size: {params.blur_kernel_size}
                    <input
                    type="range"
                    min="1"
                    max="15"
                    step="2"
                    value={params.blur_kernel_size}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, blur_kernel_size: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Blur Sigma: {params.blur_sigma}
                    <input
                    type="range"
                    min="0.1"
                    max="5"
                    step="0.1"
                    value={params.blur_sigma}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, blur_sigma: parseFloat(e.target.value) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Canny Threshold 1: {params.canny_threshold_1}
                    <input
                    type="range"
                    min="0"
                    max="255"
                    step="1"
                    value={params.canny_threshold_1}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, canny_threshold_1: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label>

                <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Canny Threshold 2: {params.canny_threshold_2}
                    <input
                    type="range"
                    min="0"
                    max="255"
                    step="1"
                    value={params.canny_threshold_2}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, canny_threshold_2: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label>

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Max Segment Angle (deg): {params.max_segment_angle_deg}
                    <input
                    type="range"
                    min="0"
                    max="180"
                    step="1"
                    value={params.max_segment_angle_deg}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, max_segment_angle_deg: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Min Segment Length: {params.min_segment_legnth}
                    <input
                    type="range"
                    min="1"
                    max="1000"
                    step="1"
                    value={params.min_segment_legnth}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, min_segment_legnth: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Segment Angle Window: {params.segment_angle_window}
                    <input
                    type="range"
                    min="1"
                    max="180"
                    step="1"
                    value={params.segment_angle_window}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, segment_angle_window: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Tile Size: {params.tile_size}
                    <input
                    type="range"
                    min="5"
                    max="40"
                    step="1"
                    value={params.tile_size}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, tile_size: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label>

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Number of Rings: {params.number_of_rings}
                    <input
                    type="range"
                    min="1"
                    max="10"
                    step="1"
                    value={params.number_of_rings}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, number_of_rings: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Initial Step Factor: {params.intiial_step_factor}
                    <input
                    type="range"
                    min="0.1"
                    max="5"
                    step="0.1"
                    value={params.intiial_step_factor}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, intiial_step_factor: parseFloat(e.target.value) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Step Size Factor: {params.step_size_factor}
                    <input
                    type="range"
                    min="0.05"
                    max="2"
                    step="0.05"
                    value={params.step_size_factor}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, step_size_factor: parseFloat(e.target.value) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Min Intersection Distance Factor: {params.min_intersection_distance_factor}
                    <input
                    type="range"
                    min="0.05"
                    max="2"
                    step="0.05"
                    value={params.min_intersection_distance_factor}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, min_intersection_distance_factor: parseFloat(e.target.value) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Max Frontiers: {params.max_frontiers}
                    <input
                    type="range"
                    min="1"
                    max="20"
                    step="1"
                    value={params.max_frontiers}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, max_frontiers: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label>

                <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Flood Fill Neighbor Points: {params.flood_fill_neighbor_points}
                    <input
                    type="range"
                    min="1"
                    max="16"
                    step="1"
                    value={params.flood_fill_neighbor_points}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, flood_fill_neighbor_points: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label>

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Flood Fill Distance Factor: {params.flood_fill_distance_factor}
                    <input
                    type="range"
                    min="0.1"
                    max="5"
                    step="0.1"
                    value={params.flood_fill_distance_factor}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, flood_fill_distance_factor: parseFloat(e.target.value) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}

                {/* <label style={{ display: 'block', marginBottom: '1rem' }}>
                    Max Background Points: {params.max_background_points}
                    <input
                    type="range"
                    min="0"
                    max="50000"
                    step="1000"
                    value={params.max_background_points}
                    onChange={(e) =>
                        setParams((prev) => ({ ...prev, max_background_points: parseInt(e.target.value, 10) }))
                    }
                    style={{ width: '100%' }}
                    />
                </label> */}
            </div>





            <canvas ref={canvasRef} />
            </div>
        </header>
        </div>
    );
}

export default App;
