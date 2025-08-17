import React, { useEffect, useRef, useState } from 'react';

import './App.css';

import CanvasDisplay from './components/CanvasDisplay';


function App() {

    const canvasRef = useRef(null);
    const canvasRef2 = useRef(null);
    const animationFrameRef = useRef(null);

    const [text, setText] = useState('');

    const [animationMode, setAnimationMode] = useState("paused");

    const maxPendingFrames = 5;
    const pendingFramesRef = useRef(0);


    const k = 100;
    const multi_step = 100;



    



    
    const default_params = {
        resize_factor: 1,
        blur_kernel_size: 3,
        blur_sigma: 1.4,
        canny_threshold_1: 50,
        canny_threshold_2: 100,
        max_segment_angle_deg: 100,
        min_segment_legnth: 20,
        segment_angle_window: 10,
        tile_size: 10,
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
    function useDebouncedValue(value, delay) {
        const [debouncedValue, setDebouncedValue] = useState(value);

        useEffect(() => {
            const timeout = setTimeout(() => setDebouncedValue(value), delay);
            return () => clearTimeout(timeout);
        }, [value, delay]);

        return debouncedValue;
    }
    const debouncedParams = useDebouncedValue(params, 50);

    const [advancedView, setAdvancedView] = useState(false); // simple (one result image / advanced -> prereqs)

    const [mosaicReady, setMosaicReady] = useState(false);

  





    // Function to progress wasm computation (if needed) and handle forward animation
    const stepK = (k) => { 

        if (!workerReady || !mosaicReady) return;
           
        if (pendingFramesRef.current < maxPendingFrames) { 
            workerRef.current.postMessage({ type: "step", data:k });
            pendingFramesRef.current++;
        }
        
    }

    // Function to handle backward animation
    const reverseStepK = (k) => { 

        if (!workerReady || !mosaicReady) return;

        if (pendingFramesRef.current < maxPendingFrames) { 
            workerRef.current.postMessage({ type: "reverse_step", data:k});
            pendingFramesRef.current++;
        }
        
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



    // CREAT REF FOR WEB WORKER
    const workerRef = useRef(null);
    const [workerReady, setWorkerReady] = useState(false);

    const handleWorkerMessage = (e) => { 
        const { type, data, error } = e.data;

            if (type === 'error') {
                console.error('Error from worker:', error);
            }

            if (type === 'wasm_ready') {
                console.log('✅ WASM module ready');
                setWorkerReady(true);
            }

            if (type === "mosaic_created") { 
                const { width, height } = data;
                const size_string = "Mosaic: " + width + " x " + height;
                setText(size_string);
                console.log("mosaic creation sucessful, message received in main thread");

                setMosaicReady(true);

                // resize canvas
                // const canvas = canvasRef.current;
                // canvas.width = width;
                // canvas.height = height;
                // clearOnscreenCanvas();
            }


            if (type === "frame") { 
                const { width, height, pixels } = e.data;

                const canvas = canvasRef.current;
                const ctx = canvas.getContext('2d');

                // if we are getting a new size, reset the canvas
                if (canvas.width !== width || canvas.height !== height) {
                    canvas.width = width;
                    canvas.height = height;
                    clearOnscreenCanvas();
                }
                
                const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);

                // console.log("main thread received frame");
                
                ctx.putImageData(imageData, 0, 0);
                pendingFramesRef.current--;
            }

            // TODO merge into one frame call widh dest canvas as worker-sent data member
            if (type == "contours") {

                if (canvasRef2.current === null) return;

                const { width, height, pixels } = e.data;

                const canvas = canvasRef2.current;
                const ctx = canvas.getContext('2d');

                // if we are getting a new size, reset the canvas
                if (canvas.width !== width || canvas.height !== height) {
                    canvas.width = width;
                    canvas.height = height;
                    ctx.fillStyle = "black";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                }
                
                const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
                
                ctx.putImageData(imageData, 0, 0);
            }

    };
    





    useEffect(() => {
        const worker = new Worker(`${process.env.PUBLIC_URL}/wasmWebWorker.js`);
        

        worker.onmessage = handleWorkerMessage;
            

        workerRef.current = worker;

        return () => worker.terminate();
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
    }, [animationMode]);




    useEffect(() => {
        if (!workerReady) return;

        workerRef.current.postMessage({ type: "clear_data" });
        workerRef.current.postMessage({ type: "set_parameters", data: debouncedParams });
        workerRef.current.postMessage({ type: "run_contour_pipeline" });


    }, [debouncedParams]);

    






    const getWindowSize = () => {
        const window_width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
        const window_height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
        return { window_width, window_height };
    }
    const { window_width, window_height } = getWindowSize();
    const canvas_display_width = 0.75 * window_width;
    const canvas_display_height = 0.60 * window_height;
    // console.log("canvas element dimensions: ", canvas_display_width, " x ", canvas_display_height);


    const scaleCanvas = (canvas, scaled_width, scaled_height) => {
        if (!canvas) return null;

        // Resize canvas element and style
        canvas.width = scaled_width;
        canvas.height = scaled_height;
        canvas.style.width = `${scaled_width}px`;
        canvas.style.height = `${scaled_height}px`;

        // Fill with background color
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;

        ctx.fillStyle = "blue";
        ctx.fillRect(0, 0, scaled_width, scaled_height);

        // return { width: scaledWidth, height: scaledHeight };
    };



    const handleUpload = async (event) => {
        const file = event.target.files[0];
        if (!file || !workerRef.current || !workerReady) return;

        stopAnimation();
        setMosaicReady(false); // will be reset once the worker responds

        // Read image into bytes
        const arrayBuffer = await file.arrayBuffer();
        const byteArray = new Uint8Array(arrayBuffer);

        // Send message to worker with image + parameters
        workerRef.current.postMessage({
            type: 'handle_image_upload',
            data: {
                bytes: byteArray,
                parameters: params, // This must be a plain JS object matching your embind Parameters
            },
        });

        workerRef.current.postMessage({ type: "run_contour_pipeline" });


        
        // Resize displays based on aspect ratio of uploaded image
        const img = new Image();
        img.src = URL.createObjectURL(file);

        img.onload = () => {


            const original_width = img.width;
            const original_height = img.height;



            const max_width = canvas_display_width;
            const max_height = canvas_display_height;
            

            // Compute scale preserving aspect ratio
            const scale = Math.min(max_width / original_width, max_height / original_height);
            const scaled_width = Math.round(original_width * scale);
            const scaled_height = Math.round(original_height * scale);

            scaleCanvas(canvasRef.current, scaled_width, scaled_height);
            scaleCanvas(canvasRef2.current, scaled_width, scaled_height);

            // Clean up object URL
            URL.revokeObjectURL(img.src);
        };



    };


    /*
    --------------------------------------
        vvv FUNCTIONS FOR BUTTONS/IO vvv
    --------------------------------------
    */

    const handleWorkerTest = () => { 
        if (!workerReady) { 
            console.log("wasm not ready yet");
            return;
        }
        // workerRef.current.postMessage({ type: "ping" })
        workerRef.current.postMessage({ type: "step", data:k })
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
        if (!workerReady) return;

        workerRef.current.postMessage({ type: 'reset_pointer'});
        clearOnscreenCanvas();
       
        setAnimationMode("play");
    }

    const handleReset = () => { 

        setAnimationMode("paused");
        workerRef.current.postMessage({ type: 'reset_pointer'});
        clearOnscreenCanvas();
        
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

            <p></p>
            <button onClick={handleWorkerTest}>
                test worker
            </button>
            <button onClick={() => setAdvancedView(!advancedView)}>
                Toggle Advanced View
            </button>


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

            </div>



            {/* <div style={{
                width: '1200px',        // Half the screen width
                maxWidth: '100%',     // Prevents overflow
                border: '1px solid #ccc',  // Optional visual border
                position: 'relative',      // Useful if canvases use absolute positioning
                overflow: 'hidden',        // Hides overflow if any canvas exceeds bounds
            }}>

                

                <canvas
                    ref={canvasRef2}
                    style={{ display: advancedView ? 'block' : 'none' }}
                />

                <canvas 
                    style={{ display: 'block' }}
                    ref={canvasRef} 
                />

            </div> */}



            <CanvasDisplay ref={canvasRef} width={canvas_display_width} height={canvas_display_height} />
            {/* <CanvasDisplay ref={canvasRef2} width={800} height={600} style={{ display: advancedView ? 'block' : 'none' }} /> */}

            

            </div>
        </header>
        </div>
    );
}

export default App;