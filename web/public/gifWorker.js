// /public/gifWorker.js

importScripts('/libs/gif.js', '/libs/gif.worker.js');


let wasmInstance = null;
let wasmIsReady = false; // if called but not ready -> error is thrown
let mosaic = null;
let total_tiles_estimate = -1;



// import the compiled glue code, which will read self.Module
importScripts('/wasm/image_module.js');



// Create the module instance by calling the factory function
Module({
  locateFile: (path) => {
    if (path.endsWith('.wasm')) {
      return '/wasm/image_module.wasm';
    }
    return path;
  }
}).then((instance) => {

    // Save instance for later
    wasmInstance = instance;
    wasmIsReady = true;
    self.postMessage({ type: 'gif_wasm_ready' }); // TODO catch this in app.js

}).catch(err => {
    console.error("gif WASM module instantiation failed:", err);
    self.postMessage({ type: 'wasm-error', error: err.message });
});



self.onmessage = function (e) {
    const { type, data } = e.data;
    
    if (!wasmIsReady) { 
        self.postMessage({ type: "wasm_load_error", error:"worked called but wasm not ready" });
    }

    else if (type === "handle_image_upload") { 

        const { bytes, parameters } = data;

        try {

            console.log("handling image upload in gif worker");

            if (mosaic) { 
                mosaic.delete();
                mosaic = null;
                computationComplete = false;
                // console.log("deleting existing mosaic");
            }

            // Create Mosaic
            const params = createParamsObject(parameters);
            mosaic = new wasmInstance.Mosaic(params);

            
            // Load image data into Mosaic
            const byteArray = new Uint8Array(bytes);
            loadMosaicFromBytes(byteArray, byteArray.length);
            // console.log("worker received image of length: ", byteArray.length);
            // console.log("worker has created mosaic successfully");

            const size = mosaic.size();
            console.log("gif mosaic.size(): ", size);

            // Post message when animation is completed
            // self.postMessage({
            //     type: 'gif_mosaic_created',
            //     data: {
            //         width: size.width,
            //         height: size.height,
            //     },
            // });

        } catch (err) {
            console.error('Error creating mosaic in worker:', err);
            self.postMessage({
                type: 'error',
                error: 'Could not create mosaic in gif worker',
            });
        }

    }

    else if (type === "set_parameters") { 
        if (!mosaic) { 
            self.postMessage({
                type: "error",
                error: "set params called but no mosaic exists in gif worker"
            });
            return;
        }
        mosaic.clearData();
        const params = createParamsObject(data);
        mosaic.setParameters(params);
        mosaic.resizeOriginal(); // TODO this can be refactored on c++ side


        const { width, height } = mosaic.size();
        const n = parseInt(params.tile_size);
        const packingDensity = 0.60; // could adjust based on empirical data
        total_tiles_estimate = Math.floor(packingDensity * (width * height) / (n * n));

    }

    else if (type === "create_gif") { 
        const { k, delay, end_time } = data;

        if (!mosaic) { 
            self.postMessage({
                type: "error",
                error: "create gif called but no mosaic exists"
            });
            return;
        }

        try {
            mosaic.setRenderPointer(0);
            const { width, height } = mosaic.size();
            console.log("creating gif size: ", width, height);

            const gif = new GIF({
                workers: 10,
                quality: 10,
                width,
                height,
                workerScript: '/libs/gif.worker.js'
            });

            let step_valid = true;
            let tiles_placed = 0;
            let progress = 0;

            // const delay = 100; // ms per frame



            while (step_valid) {
                step_valid = mosaic.stepK(k);

                const current = mosaic.getRenderPointer();
                mosaic.renderImageRange(0, current);
                mosaic.setRenderPointer(current + k);
                const { pixels } = getImageBuffer(mosaic.getCanvasPtr());
                


                // Create ImageData for the frame
                const imageData = new ImageData(pixels, width, height);
                gif.addFrame(imageData, { delay });

                tiles_placed += k;
                progress = (tiles_placed / total_tiles_estimate) * 100;
                // console.log("progress estimate: ", progress, "%")
                self.postMessage({type: "gif_progress", data:progress});

            }



            const pause_time_ms = 1000 * end_time;
            const pause_frames = pause_time_ms / delay;

            const { pixels } = getImageBuffer(mosaic.getCanvasPtr());
            const imageData = new ImageData(pixels, width, height);

            for (let i = 0; i < pause_frames; i++) { 
                gif.addFrame(imageData, { delay });
            }



            console.log("gif computation stopped after: ", tiles_placed, "steps");

            gif.on('finished', function(blob) {
                console.log("✅ GIF encoding finished, blob size:", blob.size, "bytes");
                self.postMessage({ type: 'gif_ready', blob });
            });

            gif.render();
        } catch (err) {
            console.error("Error during gif creation:", err);
            self.postMessage({ type: "error", error: "gif creation failed: " + err.message });
        }
    }


}





function createParamsObject(user_params) { 
    // take js object and create Params object from module

    const params = new wasmInstance.Parameters;
    params.resize_factor = user_params.resize_factor;
    params.blur_kernel_size = user_params.blur_kernel_size;
    params.blur_sigma = user_params.blur_sigma;
    params.canny_threshold_1 = user_params.canny_threshold_1;
    params.canny_threshold_2 = user_params.canny_threshold_2;
    params.max_segment_angle_rad = user_params.max_segment_angle_deg * Math.pi / 180.0; 
    params.min_segment_length = user_params.min_segment_legnth;
    params.segment_angle_window = user_params.segment_angle_window;
    params.tile_size = user_params.tile_size;
    params.number_of_rings = user_params.number_of_rings;
    params.initial_step = user_params.intiial_step_factor * user_params.tile_size;
    params.step_size = user_params.step_size_factor * user_params.tile_size;
    params.min_intersection_distance = user_params.min_intersection_distance_factor * user_params.tile_size;
    params.max_frontiers = user_params.max_frontiers;
    params.flood_fill_neighbor_points = user_params.flood_fill_neighbor_points;
    params.distance_from_center = user_params.flood_fill_distance_factor * user_params.tile_size;
    params.random_background_points = user_params.max_background_points;
    params.canny_resize_factor = user_params.canny_resize_factor;

    return params;
}


function loadMosaicFromBytes(byteArray, size) {
    if (!mosaic) return;

    // Allocate memory on Emscripten heap
    const ptr = wasmInstance._malloc(byteArray.length);
    if (!ptr) {
        throw new Error("Failed to allocate memory");
    }

    try {
        // Copy data to heap
        wasmInstance.HEAPU8.set(byteArray, ptr);
        
        // Call the function with the pointer
        mosaic.loadImageFromHeap(ptr, size);
        console.log("loadImageFromHeap successful");

    } catch (err) {
        console.error('Error loading image from heap in worker:', err);
        self.postMessage({
            type: 'error',
            error: 'Could not create mosaic',
        });
    } finally {
        // Always free the memory
        wasmInstance._free(ptr);
    }

}





function getBufferLength() {
    const { width, height } = mosaic.size();
    return width * height * 4;
}

function getImageBuffer(image_ptr) { 
    if (!image_ptr) {
        throw new Error("No data available");
    }
    
    const data_ptr = image_ptr.getRawData();
    if (!data_ptr) {
        throw new Error("Failed to get raw data pointer");
    }
    
    const width = image_ptr.getWidth();
    const height = image_ptr.getHeight();

    // Make a copy that is safe to transfer
    const total_size = width * height * 4;
    const rawView = new Uint8ClampedArray(wasmInstance.HEAPU8.buffer, data_ptr, total_size);
    const copy = new Uint8ClampedArray(rawView); // OR rawView.slice()

    

    return {
        width,
        height,
        pixels: copy
    };
}