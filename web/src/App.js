import React, { useEffect, useRef, useState } from 'react';

import './App.css';

import CanvasDisplay from './components/CanvasDisplay';
import GridCanvasDisplay from './components/GridCanvasDisplay';
import SliderInput from './components/SliderInput';


function App() {

    const canvasRef = useRef(null); // output
    const canvasRef2 = useRef(null); // canny
    const canvasRef3 = useRef(null); // original
    const canvasRef4 = useRef(null); // debug
    const refs_array = useRef([canvasRef3, canvasRef2, canvasRef4, canvasRef]);

    const animationFrameRef = useRef(null);

    const [text, setText] = useState('');

    const [animationMode, setAnimationMode] = useState("paused");

    const maxPendingFrames = 3;
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
        canny_resize_factor: 1,
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
    const advancedViewRef = useRef(false);

    const [mosaicReady, setMosaicReady] = useState(false);



    const [pngUrl, setPngUrl] = useState(null);

    const [uploadedFilename, setUploadedFilename] = useState(null);

  





    // Function to progress wasm computation (if needed) and handle forward animation
    const stepK = (k) => { 

        if (!workerReady || !mosaicReady) return;
           
        if (pendingFramesRef.current < maxPendingFrames) { 
            // console.log("main thread requesting frame, advanced view: ", advancedViewRef.current);
            workerRef.current.postMessage({ 
                type: "step", 
                data: {
                    k: k,
                    advancedView: advancedViewRef.current
                }
            });
            pendingFramesRef.current++;

            // if (advancedView) { 
            //     workerRef.current.postMessage({type: "get_debug_image"});
            // }
        }

       
        
    }

    // Function to handle backward animation
    const reverseStepK = (k) => { 

        if (!workerReady || !mosaicReady) return;

        if (pendingFramesRef.current < maxPendingFrames) { 
            workerRef.current.postMessage({ 
                type: "reverse_step", 
                data: {
                    k: k,
                    advancedView: advancedViewRef.current
                }
            });
            pendingFramesRef.current++;
        }
        
    }

    const clearOnscreenCanvas = (canvasRef) => {


       if (!canvasRef) return;

        const canvas = canvasRef.current;
        if (canvas) { 
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
        


 
        
    }

    const stopAnimation = () => { 

        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

        setAnimationMode("paused");
    }

    function imageDataToBlob(imageData) {
        const canvas = document.createElement('canvas');
        canvas.width = imageData.width;
        canvas.height = imageData.height;

        const ctx = canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);

        return new Promise(resolve => {
            canvas.toBlob(blob => {
                resolve(blob);
            }, 'image/png');
        });
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
                // console.log("mosaic creation sucessful, message received in main thread");

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

                // console.log('main thread recieves contours_image post message');

                if (canvasRef2.current === null) return;

                

                const { width, height, pixels } = e.data;
                // console.log("main thread receives contour image dim:", width, " x " , height);

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

                // console.log("putting contours image data to canvasRef2");
                
            }


            if (type == "original") {

                // console.log('main thread receives original post messge');

                if (canvasRef3.current === null) return;

                

                const { width, height, pixels } = e.data;
                // console.log("main thread recieves original image dim:", width, " x " , height);

                const canvas = canvasRef3.current;
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

                // console.log("putting contours image data to canvasRef3");
                
            }

            if (type == "debug_image") {

                // console.log('main thread receives debug post messge');

                if (canvasRef4.current === null) return;

                

                const { width, height, pixels } = e.data;
                // console.log("main thread recieves original image dim:", width, " x " , height);

                const canvas = canvasRef4.current;
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

                // console.log("putting contours image data to canvasRef3");
                
            }


            if (type === "output_image") { 
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


            }

            if (type === "final_output_image") { 
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


                // save for later download
                imageDataToBlob(imageData).then(blob => {
                    const url = URL.createObjectURL(blob);
                    setPngUrl(url);
                });


            }


    };


    const gifWorkerRef = useRef(null);
    const [gifWorkerReady, setGifWorkerReady] = useState(false);
    const [gifUrl, setGifUrl] = useState(null);
    const [gifProgress, setGifProgress] = useState(0);

    const handleGifWorkerMessage = (e) => {
        const { type, data, error, blob } = e.data;

        if (type === 'error') {
            console.error('❌ Error from worker:', error);
        }

        if (type === 'gif_wasm_ready') {
            console.log('✅ gif WASM module ready');
            setGifWorkerReady(true);
        }

        if (type === "gif_progress") { 
            setGifProgress(data);
        }

        if (type === 'gif_ready') {
            console.log('🎞️ Received gif blob from worker');
            setGifProgress(100);
            
            // Revoke previous URL if it exists to avoid memory leaks
            if (gifUrl) {
                URL.revokeObjectURL(gifUrl);
            }

            // Create a new URL for the blob and update state
            const url = URL.createObjectURL(blob);
            setGifUrl(url);
        }
    };

    





    useEffect(() => {

        const worker = new Worker(`${process.env.PUBLIC_URL}/wasmWebWorker.js`);
        const gif_worker = new Worker(`${process.env.PUBLIC_URL}/gifWorker.js`);

        worker.onmessage = handleWorkerMessage;
        gif_worker.onmessage = handleGifWorkerMessage;
            

        workerRef.current = worker;
        gifWorkerRef.current = gif_worker;

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
        if (workerReady) {

            workerRef.current.postMessage({ type: "clear_data" });
            workerRef.current.postMessage({ type: "set_parameters", data: debouncedParams });
            workerRef.current.postMessage({ type: "run_contour_pipeline" });
        }

        

  


    }, [debouncedParams]);

    






    const getWindowSize = () => {
        const window_width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
        const window_height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
        return { w: window_width, h: window_height };
    }

    const getDisplaySize = () => {
        const { w, h } = getWindowSize();
        return { w: 0.75 * w, h: 0.7 * h};
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
    const [uploadedImage, setUploadedImage] = useState(null);
    const handleUpload = async (event) => {
        const file = event.target.files[0];
        if (!file || !workerRef.current || !workerReady) return;

        stopAnimation();
        setMosaicReady(false); // will be reset once the worker responds


        // save file name
        setUploadedFilename(file.name);

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

        workerRef.current.postMessage({ type: "run_contour_pipeline" }); // TODO is this necessary?

        // We don't use useEffect with image because of the dependencies 
        gifWorkerRef.current.postMessage({
            type: 'handle_image_upload',
            data: {
                bytes: byteArray,
                parameters: params, // This must be a plain JS object matching your embind Parameters
            },
        });



        
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



        }
    }

    // TOGGLE ADVANCED VIEW
    useEffect(() => {
        if (!mosaicReady) return;

        handleScaling(uploadedImageSize.w, uploadedImageSize.h);

        if (advancedView) {


            workerRef.current.postMessage({type: "set_debug_mode", data: true});
            advancedViewRef.current = true;
            


            let attempts = 0;
            const maxAttempts = 20; // 20 * 50ms = 1s max wait
            const interval = setInterval(() => {
                if (canvasRef.current && canvasRef3.current && canvasRef2.current && canvasRef4.current) {
                    clearInterval(interval);
                    workerRef.current.postMessage({ type: "get_contour_image" });
                    workerRef.current.postMessage({ type: "get_original_image" });
                    workerRef.current.postMessage({ type: "get_debug_image" });
                    workerRef.current.postMessage({ type: "get_output_image" });

                  
                } else {
                    attempts++;
                    if (attempts > maxAttempts) {
                        console.warn("canvas components never became available.");
                        clearInterval(interval);
                    }
                }
                
            }, 50); // check every 50ms

            // Clean up on unmount or if advancedView becomes false
            return () => clearInterval(interval);
        }
        else { 
            advancedViewRef.current = false;
            workerRef.current.postMessage({type: "set_debug_mode", data: false});

            let attempts = 0;
            const maxAttempts = 20; // 20 * 50ms = 1s max wait
            const interval = setInterval(() => {
                if (canvasRef.current) {
                    clearInterval(interval);
                    workerRef.current.postMessage({ type: "get_output_image" });

                  
                } else {
                    attempts++;
                    if (attempts > maxAttempts) {
                        console.warn("canvas components never became available.");
                        clearInterval(interval);
                    }
                }
                
            }, 50); // check every 50ms

            // Clean up on unmount or if advancedView becomes false
            return () => clearInterval(interval);
        }

    }, [advancedView, mosaicReady, debouncedParams]);



    /*
    --------------------------------------
        vvv FUNCTIONS FOR BUTTONS/IO vvv
    --------------------------------------
    */

    const handleWorkerTest = () => { 



        console.log("uploaded file name: ", uploadedFilename);
        console.log("stripped: ", uploadedFilename.replace(/\.[^/.]+$/, ""));
    }


    const handlePlay = () => { 
        // console.log("play button pressed");
        // console.log("animationMode before change: ", animationMode);
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
        clearOnscreenCanvas(canvasRef);
        clearOnscreenCanvas(canvasRef4);
       
        setAnimationMode("play");
    }

    const handleReset = () => { 

        setAnimationMode("paused");
        workerRef.current.postMessage({ type: 'reset_pointer'});
        clearOnscreenCanvas(canvasRef);
        clearOnscreenCanvas(canvasRef4);
        
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


    const handleImageExport = () => { 

        setPngUrl(null);

        if (workerReady) {

            workerRef.current.postMessage({ type: "set_parameters", data: debouncedParams });
            workerRef.current.postMessage({ type: "get_final_output_image" })
        }
    }


    const [gifPause, setGifPause] = useState(1);
    const [gifDelay, setGifDelay] = useState(50);
    const [stepsPerFrame, setStepsPerFrame] = useState(k);
    const handleGenerateGif = () => { 

        setGifUrl(null);

        if (gifWorkerReady) {



            gifWorkerRef.current.postMessage({ type: "set_parameters", data: debouncedParams });
            gifWorkerRef.current.postMessage({
                type: "create_gif",
                data: {
                    k: stepsPerFrame,
                    delay: gifDelay,
                    end_time: gifPause,
                }
            })
        }
    }

    const getOutputFilename = (extension) => { 
        if (!uploadedFilename) return;
        return uploadedFilename.replace(/\.[^/.]+$/, "") + "_mosaic" + extension;
    }

    return (
        <div className="App">
            <header className="app-layout">
            {/* Left Sidebar */}
            <div className="sidebar">

                <h1>Mosaic Filter</h1>

                {/* Upload Section */}
                <section>
                <h3>Upload</h3>
                <label htmlFor="file-upload" className="file-upload-label">Choose File</label>
                <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleUpload}
                    className="file-input"
                />
                {uploadedFilename && (
                    <p>{uploadedFilename}</p>
                )}
                </section>


                {/* Animation Controls */}
                <section>
                <h3>Animation</h3>
                <button onClick={handlePlay}>Play</button>
                <button onClick={handlePause}>Pause</button>
                <button onClick={handleReverse}>Reverse</button>
                <button onClick={handleReplay}>Replay</button>
                <button onClick={handleReset}>Reset</button>
                <button onClick={stepForward}>Step Forward</button>
                <button onClick={stepBack}>Step Back</button>
                <button onClick={stepForwardMulti}>Step Forward Multi</button>
                <button onClick={stepBackMulti}>Step Back Multi</button>
                </section>

                {/* View Controls */}
                <section>
                <h3>View</h3>
                <button onClick={() => setAdvancedView(!advancedView)}>Toggle Advanced View</button>
                </section>

                {/* Parameter Tuning */}
                <section>
                <h3>Tune Parameters</h3>
                <SliderInput
                    label="Resize Factor"
                    value={params.resize_factor}
                    min={0.1}
                    max={5}
                    step={0.1}
                    onChange={(newVal) => setParams((prev) => ({ ...prev, resize_factor: newVal }))}
                />

                <SliderInput
                    label="Canny Resize Factor"
                    value={params.canny_resize_factor}
                    min={0.1}
                    max={5}
                    step={0.1}
                    onChange={(newVal) => setParams((prev) => ({ ...prev, canny_resize_factor: newVal }))}
                />

                <SliderInput
                    label="Canny Threshold 1"
                    value={params.canny_threshold_1}
                    min={0}
                    max={255}
                    step={1}
                    onChange={(newVal) => setParams((prev) => ({ ...prev, canny_threshold_1: newVal }))}
                />

                <SliderInput
                    label="Canny Threshold 2"
                    value={params.canny_threshold_2}
                    min={0}
                    max={255}
                    step={1}
                    onChange={(newVal) => setParams((prev) => ({ ...prev, canny_threshold_2: newVal }))}
                />

                <SliderInput
                    label="Tile Size"
                    value={params.tile_size}
                    min={5}
                    max={40}
                    step={1}
                    onChange={(newVal) => setParams((prev) => ({ ...prev, tile_size: newVal }))}
                />

                <SliderInput
                    label="Max Frontiers"
                    value={params.max_frontiers}
                    min={1}
                    max={20}
                    step={1}
                    onChange={(newVal) => setParams((prev) => ({ ...prev, max_frontiers: newVal }))}
                />

                <SliderInput
                    label="Flood Fill Neighbor Points"
                    value={params.flood_fill_neighbor_points}
                    min={1}
                    max={16}
                    step={1}
                    onChange={(newVal) => setParams((prev) => ({ ...prev, flood_fill_neighbor_points: newVal }))}
                />
                </section>


                {/* Save Options */}
                <section>
                <h3>Save Image</h3>
                <button onClick={handleImageExport}>Generate PNG</button>
                {/* PNG Output */}
                {pngUrl && (
                <div>

                    <img src={pngUrl} alt="Generated PNG" style={{ maxWidth: '100px', height: 'auto' }} />

                    <div>
                    <a href={pngUrl} download={getOutputFilename(".png")}>
                    <button>Download PNG</button>
                    </a>
                    <button onClick={() => window.open(pngUrl, '_blank')} style={{ marginLeft: '10px' }}>
                    View PNG in New Tab
                    </button>
                    </div>
                </div>
                )}
                </section>

                {/* Save Options */}
                <section>
                <h3>Save Animation</h3>
                

                <SliderInput
                    label="Tiles Per Frame"
                    value={stepsPerFrame}
                    min={1}
                    max={1000}
                    step={10}
                    onChange={(newVal) => setStepsPerFrame(Math.round(newVal))}
                />

                <SliderInput
                    label="Gif Frame Time ms"
                    value={gifDelay}
                    min={35}
                    max={500}
                    step={5}
                    onChange={(newVal) => setGifDelay(Math.round(newVal))}
                />

                <SliderInput
                    label="Seconds of Pause at End"
                    value={gifPause}
                    min={0.1}
                    max={5}
                    step={0.1}
                    onChange={setGifPause}
                />

                <button onClick={handleGenerateGif}>Generate GIF</button>

                {/* GIF Output */}
                {(gifProgress > 0 && !gifUrl) && <p>Loading... {Number(gifProgress).toFixed(2)}%</p>}
                {gifUrl && (
                    <div>
                        <img src={gifUrl} alt="Generated GIF" style={{ maxWidth: '100px', height: 'auto' }} />

                        <div>  {/* <-- wrap buttons here */}
                        <a href={gifUrl} download={getOutputFilename(".gif")}>
                            <button>Download GIF</button>
                        </a>
                        <button onClick={() => window.open(gifUrl, '_blank')} style={{ marginLeft: '10px' }}>
                            View GIF in New Tab
                        </button>
                        </div>
                    </div>
                )}
                </section>

                {/* Developer Tools */}
                <section>
                <h3>Developer</h3>
                <button onClick={handleWorkerTest}>Test Worker</button>
                <p>{text || "Click the button to run C++ code"}</p>
                </section>
            </div>

            {/* Main Canvas Area */}
            <div className="canvas-area">
                {advancedView ? (
                <GridCanvasDisplay
                    canvasRefs={refs_array.current}
                    size={canvasSize}
                    gridLayout={gridLayout}
                />
                ) : (
                <CanvasDisplay ref={canvasRef} size={canvasSize} />
                )}
            </div>
            </header>
        </div>
    );


}

export default App;