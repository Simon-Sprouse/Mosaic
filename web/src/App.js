import React, { useEffect, useRef, useState } from 'react';

import './App.css';

import CanvasDisplay from './components/CanvasDisplay';
import GridCanvasDisplay from './components/GridCanvasDisplay';


function App() {

    const canvasRef = useRef(null);
    const canvasRef2 = useRef(null);
    const canvasRef3 = useRef(null);
    const refs_array = useRef([canvasRef2, canvasRef, canvasRef3]);

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

                setAnimationMode("play");

                
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

                console.log('main thread recieves contours_image');

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

                console.log("putting contours image data to canvasRef2");
                
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
        return { w: window_width, h: window_height };
    }

    const getDisplaySize = () => {
        const { w, h } = getWindowSize();
        return { w: 0.75 * w, h: 0.6 * h};
    }


    const findOptimalRectanglePacking = (n, tableWidth, tableHeight, rectWidth, rectHeight) => {
        if (n <= 0 || tableWidth <= 0 || tableHeight <= 0 || rectWidth <= 0 || rectHeight <= 0) {
            throw new Error('All parameters must be positive numbers');
        }
        
        let bestLayout = null;
        let maxScale = 0;
        
        // Try all possible grid configurations
        // For n rectangles, we need rows * cols >= n
        for (let rows = 1; rows <= n; rows++) {
            // Calculate minimum columns needed for this row count
            const cols = Math.ceil(n / rows);
            
            // Calculate the maximum scale factor for this grid configuration
            const scaleX = tableWidth / (cols * rectWidth);   // Scale limited by width
            const scaleY = tableHeight / (rows * rectHeight); // Scale limited by height
            
            // Use the smaller scale factor to ensure rectangles fit in both dimensions
            const scale = Math.min(scaleX, scaleY);
            
            // Skip if this configuration doesn't fit at all
            if (scale <= 0) continue;
            
            // Check if this is the best configuration so far
            if (scale > maxScale) {
                maxScale = scale;
                bestLayout = {
                    rows: rows,
                    cols: cols,
                    scale: scale,
                    scaledRectWidth: rectWidth * scale,
                    scaledRectHeight: rectHeight * scale,
                    totalUsedWidth: cols * rectWidth * scale,
                    totalUsedHeight: rows * rectHeight * scale,
                    unusedRectangles: rows * cols - n
                };
            }
        }
        
        if (!bestLayout) {
            throw new Error('No valid packing configuration found - rectangles may be too large for the table');
        }
        
        return bestLayout;
    }


    const [uploadedImageSize, setUploadedImageSize] = useState({ w: 0, h: 0});
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

            handleScaling(img.width, img.height);
            setUploadedImageSize({ w: img.width, h: img.height });

            // Clean up object URL
            URL.revokeObjectURL(img.src);
        };

    };

    const [canvasSize, setCanvasSize] = useState(getDisplaySize());
    const [gridLayout, setGridLayout] = useState({ rows: 0, cols: 0});
    const handleScaling = (original_width, original_height) => { 

        const { w, h } = getDisplaySize();
        const max_width = w;
        const max_height = h;

        if (!advancedView) { 
            
            // Compute scale preserving aspect ratio
            const scale = Math.min(max_width / original_width, max_height / original_height);

            const scaled_width = Math.round(original_width * scale);
            const scaled_height = Math.round(original_height * scale);
            setCanvasSize({ w: scaled_width, h: scaled_height});
        }
        else {
            const grid_metadata = findOptimalRectanglePacking(refs_array.current.length, max_width, max_height, original_width, original_height);

            const scaled_width = grid_metadata.scaledRectWidth;
            const scaled_height = grid_metadata.scaledRectHeight;
            setCanvasSize({ w: scaled_width, h: scaled_height});

            const rows = grid_metadata.rows;
            const cols= grid_metadata.cols;
            setGridLayout({ rows: rows, cols: cols});

            console.log("grid metadata: ", grid_metadata);

        }
    }

    useEffect(() => {
        if (!mosaicReady) return;

        handleScaling(uploadedImageSize.w, uploadedImageSize.h);

        if (advancedView) {
            let attempts = 0;
            const maxAttempts = 20; // 20 * 50ms = 1s max wait
            const interval = setInterval(() => {
                if (canvasRef2.current) {
                    clearInterval(interval);
                    workerRef.current.postMessage({ type: "get_contour_image" });
                } else {
                    attempts++;
                    if (attempts > maxAttempts) {
                        console.warn("canvasRef2 never became available.");
                        clearInterval(interval);
                    }
                }
            }, 50); // check every 50ms

            // Clean up on unmount or if advancedView becomes false
            return () => clearInterval(interval);
        }

    }, [advancedView]);



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
        console.log("play button pressed");
        console.log("animationMode before change: ", animationMode);
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





          
            {
                advancedView ? (<GridCanvasDisplay canvasRefs={refs_array.current} size={canvasSize} gridLayout={gridLayout}/>) : 
                (<CanvasDisplay ref={canvasRef} size={canvasSize} />)
            }
            



            </div>
        </header>
        </div>
    );
}

export default App;