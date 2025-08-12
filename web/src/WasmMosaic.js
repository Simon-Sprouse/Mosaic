export default class WasmMosaic {

    constructor(Module) {

        this.Module = Module;
        this.mosaic = null;
        this.params = null;
    }
    
    createParamsObject(user_params) { 
        // take js object and create Params object from module

        const params = new this.Module.Parameters;
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

    createMosaic(user_params) {
        
        this.params = this.createParamsObject(user_params);
        this.mosaic = new this.Module.Mosaic(this.params);
        
    }

    setParameters(user_params) { 
        this.params = this.createParamsObject(user_params);
        if (this.mosaic) { 
            this.mosaic.setParameters(this.params);
        }
    }

    clearData() { 
        if (this.mosaic) { 
            this.mosaic.clearData();
        }
    }

    resizeOriginal() {
        if (this.mosaic && !this.mosaic.empty()) { 
            this.mosaic.resizeOriginal();
        }
    }

    loadMosaicFromBytes(byteArray, size) {
        if (!this.mosaic) return;

        // Allocate memory on Emscripten heap
        const ptr = this.Module._malloc(byteArray.length);
        if (!ptr) {
            throw new Error("Failed to allocate memory");
        }

        try {
            // Copy data to heap
            this.Module.HEAPU8.set(byteArray, ptr);
            
            // Call the function with the pointer
            this.mosaic.loadImageFromHeap(ptr, size);
        } finally {
            // Always free the memory
            this.Module._free(ptr);
        }

    }

    empty() {
        return this.mosaic?.empty() ?? true;
    }

    getSize() {
        const size = this.mosaic.size();
        return { width: size.width, height: size.height };
    }

    getLength() {
        const { width, height } = this.getSize();
        return width * height;
    }

    stepK(k) { 
        // this.mosaic.runAll();
        return this.mosaic.stepK(k);
        // this.mosaic.reconstructImageNewTiles();
    }


    getRenderPointer() { 
        return this.mosaic.getRenderPointer();
    }


    renderImageRange(start, num_tiles) { 
        this.mosaic.renderImageRange(start, num_tiles);
    }

    resetRenderPointer() {
        this.mosaic.setRenderPointer(0);
    }

    resetCanvas() { 
        this.mosaic.resetCanvas();
    }



    getRawData() {

        if (!this.mosaic || this.mosaic.empty()) {
            throw new Error("No mosaic data available");
        }
        
        const ptr = this.mosaic.getRawData();
        if (!ptr) {
            throw new Error("Failed to get raw data pointer");
        }
        
        const totalSize = this.getLength() * 4;
        return new Uint8ClampedArray(this.Module.HEAPU8.buffer, ptr, totalSize);
    }



    destroy() {
        if (this.mosaic) {
            this.mosaic.delete(); // Embind-managed delete
            this.mosaic = null;
        }
    }
}
