export default class WasmMosaic {
    constructor(Module) {
        this.Module = Module;
        this.params = null;
        this.create();
        
    }

    create() {
        this.params = new this.Module.Parameters;
        this.params.resize_factor = 0.5;
        this.params.blur_kernel_size = 3;
        this.params.blur_sigma = 1.4;
        this.params.canny_threshold_1 = 50;
        this.params.canny_threshold_2 = 100;
        this.params.max_segment_angle_rad = 100 * Math.pi / 180.0; // TODO why is this in rad
        this.params.min_segment_length = 20;
        this.params.segment_angle_window = 10;
        this.params.tile_size = 10;
        this.params.number_of_rings = 3;
        this.params.initial_step = 1.5 * this.params.tile_size;
        this.params.step_size = 0.25 * this.params.tile_size;
        this.params.min_intersection_distance = 0.25 * this.params.tile_size;
        this.params.max_frontiers = 4;
        this.params.flood_fill_neighbor_points = 4;
        this.params.distance_from_center = 1.5 * this.params.tile_size;
        this.params.random_background_points = 50000;
        this.params.tiles_per_frame = 20; //  TODO remove

        this.mosaic = new this.Module.Mosaic(this.params);
        
    }

    loadMosaicFromBytes(byteArray, size) {
        if (!this.mosaic) this.create();

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
