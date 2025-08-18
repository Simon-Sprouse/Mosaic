// public/wasmWebWorker.js


// vvv Debug script vvv
// fetch('/wasm/image_module.wasm').then(async (res) => {
//   const text = await res.text();
//   console.log('WASM file fetch response (as text):', text.slice(0, 100));
// }).catch(console.error);


let wasmInstance = null;
let wasmIsReady = false; // if called but not ready -> error is thrown
let mosaic = null; // stores WasmMosaic class instance
let computationComplete = false;


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
    self.postMessage({ type: 'wasm_ready' }); 

}).catch(err => {
    console.error("WASM module instantiation failed:", err);
    self.postMessage({ type: 'wasm-error', error: err.message });
});




// Listen for messages from main thread
self.onmessage = function (e) {
    const { type, data } = e.data;
    
    if (!wasmIsReady) { 
        self.postMessage({ type: "wasm_load_error", error:"worked called but wasm not ready" });
    }

    else {
        
        if (type === 'ping') {
        
            console.log("worker recieved ping message");

            const message = wasmInstance.sayHello();
            console.log("sayHello wasm returned this message to worker:", message);
            self.postMessage({ type: 'pong', data: "hello from worker"});
            self.postMessage({ type: 'pong', data: message });
        }

        
        else if (type === "handle_image_upload") { 

            const { bytes, parameters } = data;

            try {

                console.log("handling image upload");

                if (mosaic) { 
                    mosaic.delete();
                    mosaic = null;
                    computationComplete = false;
                    console.log("deleting existing mosaic");
                }

                // Create Mosaic
                const params = createParamsObject(parameters);
                mosaic = new wasmInstance.Mosaic(params);

                
                // Load image data into Mosaic
                const byteArray = new Uint8Array(bytes);
                loadMosaicFromBytes(byteArray, byteArray.length);
                console.log("worker received image of length: ", byteArray.length);
                console.log("worker has created mosaic successfully");

                const size = mosaic.size();
                console.log("mosaic.size(): ", size);

                self.postMessage({
                    type: 'mosaic_created',
                    data: {
                        width: size.width,
                        height: size.height,
                    },
                });

            } catch (err) {
                console.error('Error creating mosaic in worker:', err);
                self.postMessage({
                    type: 'error',
                    error: 'Could not create mosaic',
                });
            }

        }

        else if (type === "run_contour_pipeline") { 
            if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "pipeline called but no mosaic exists"
                });
                return;
            }
            mosaic.contourPipeline();
            const { width, height, pixels } = getImageBuffer(mosaic.getStrokesImagePtr());
            self.postMessage({
                type: 'contours',
                width,
                height,
                pixels: pixels.buffer, // send raw ArrayBuffer
            }, [pixels.buffer]);  // transfer ownership (zero-copy)
        }


        else if (type === "step") { 

            const k = data.k;
            const isAdvancedView = data.advancedView;

            if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "step called but no mosaic exists"
                });
                return;
            }

            // prevent redudant calls to wasm module
            if (!computationComplete){
                const stepValid = mosaic.stepK(k);
                if (!stepValid) computationComplete = true;
            }
            



            const current = mosaic.getRenderPointer();
            mosaic.renderImageRange(current, k);
            mosaic.setRenderPointer(current + k);

            const { width, height, pixels } = getImageBuffer(mosaic.getCanvasPtr());
            self.postMessage({
                type: 'frame',
                width,
                height,
                pixels: pixels.buffer, // send raw ArrayBuffer
            }, [pixels.buffer]);  // transfer ownership (zero-copy)



            if (isAdvancedView) { 
                const { width, height, pixels } = getImageBuffer(mosaic.getDebugCanvasPtr());
                self.postMessage({
                    type: 'debug_image',
                    width,
                    height,
                    pixels: pixels.buffer, // send raw ArrayBuffer
                }, [pixels.buffer]);  // transfer ownership (zero-copy)
            }

        }

        else if (type === "reverse_step") { 

            const k = data.k;
            const isAdvancedView = data.advancedView;

            if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "reverse step called but no mosaic exists"
                });
                return;
            }

            const current = mosaic.getRenderPointer();
            mosaic.resetCanvas();
            mosaic.setRenderPointer(0);
            const num_steps = Math.max(0, current - k);
            mosaic.renderImageRange(0, num_steps);
            mosaic.setRenderPointer(num_steps);
        
            const { width, height, pixels } = getImageBuffer(mosaic.getCanvasPtr());

            self.postMessage({
                type: 'frame',
                width,
                height,
                pixels: pixels.buffer, // send raw ArrayBuffer
            }, [pixels.buffer]);  // transfer ownership (zero-copy)



            if (isAdvancedView) { 
                const { width, height, pixels } = getImageBuffer(mosaic.getDebugCanvasPtr());
                self.postMessage({
                    type: 'debug_image',
                    width,
                    height,
                    pixels: pixels.buffer, // send raw ArrayBuffer
                }, [pixels.buffer]);  // transfer ownership (zero-copy)
            }


        }

        else if (type === "reset_pointer") { 

            if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "reset pointer called but no mosaic exists"
                });
                return;
            }

            mosaic.setRenderPointer(0);
            mosaic.resetCanvas();
        }

        else if (type === "set_debug_mode") { 
            if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "reset pointer called but no mosaic exists"
                });
                return;
            }

            mosaic.setDebugMode(data);
        }


        else if (type === "clear_data") { 
             if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "clear data called but no mosaic exists"
                });
                return;
            }
            mosaic.clearData();
            computationComplete = false;
        }

        else if (type === "set_parameters") { 
             if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "set params called but no mosaic exists"
                });
                return;
            }
            const params = createParamsObject(data);
            mosaic.setParameters(params);
            mosaic.resizeOriginal(); // TODO this can be refactored on c++ side
        }


        else if (type === "get_contour_image") { 
            if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "get contour image called but no mosaic exists"
                });
                return;
            }
            const { width, height, pixels } = getImageBuffer(mosaic.getStrokesImagePtr());
            self.postMessage({
                type: 'contours',
                width,
                height,
                pixels: pixels.buffer, // send raw ArrayBuffer
            }, [pixels.buffer]);  // transfer ownership (zero-copy)
        }

        else if (type === "get_original_image") { 
            console.log("worker gets request for original image");
            if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "get original image called but no mosaic exists"
                });
                return;
            }
            
            const { width, height, pixels } = getImageBuffer(mosaic.getOriginalImagePtr());
            self.postMessage({
                type: 'original',
                width,
                height,
                pixels: pixels.buffer, // send raw ArrayBuffer
            }, [pixels.buffer]);  // transfer ownership (zero-copy)
            console.log("worker sending original image");
        }

        else if (type === "get_debug_image") { 

            if (!mosaic) { 
                self.postMessage({
                    type: "error",
                    error: "get original image called but no mosaic exists"
                });
                return;
            }
            
            const current = mosaic.getRenderPointer();
            mosaic.renderDebugImageRange(0, current);
            const { width, height, pixels } = getImageBuffer(mosaic.getDebugCanvasPtr());
            self.postMessage({
                type: 'debug_image',
                width,
                height,
                pixels: pixels.buffer, // send raw ArrayBuffer
            }, [pixels.buffer]);  // transfer ownership (zero-copy)
            console.log("worker sending debug image");

        }



    }

};


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