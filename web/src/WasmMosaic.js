export default class WasmMosaic {

    constructor(Module) {
        this.Module = Module;
        this.mosaic_ptr = null;
    }



    loadMosaicFromBytes(byteArray) { 

        if (this.mosaic_ptr) {
            this.Module._deleteMosaic(this.mosaic_ptr); // ✅ free old image
            this.mosaic_ptr = null;
        }
        
        const length = byteArray.length;
        const ptr = this.Module._malloc(length);
        this.Module.HEAPU8.set(byteArray, ptr);

        this.mosaic_ptr = this.Module._loadMosaicFromBytes(ptr, length);  // assumes C++ function exists

        this.Module._free(ptr);
    }

    empty() {
        if (!this.mosaic_ptr) return true;  // No image loaded yet = empty
        return !!this.Module._mosaicIsEmpty(this.mosaic_ptr);  // Cast C++ bool (0/1) to JS bool
    }

    getSize() {
        const string_ptr = this.Module._getMosaicSizeStr(this.mosaic_ptr);
        const size_string = this.Module.UTF8ToString(string_ptr);
        const [width, height] = size_string.split(',').map(Number);
        return { width, height };
    }

    getLength() { 
        const { width, height } = this.getSize();
        return width * height;
    }

    runAndGetData() {

    
        if (!this.Module) {
            throw new Error("Module not available")
        }
        if (!this.Module.HEAPU8) {
            throw new Error("Module.HEAPU8 not available");
          }
      
        const data_ptr = this.Module._getMosaicOutput(this.mosaic_ptr);
        const total_size = this.getLength() * 4; // assuming RGBA
      
        if (typeof data_ptr !== 'number' || data_ptr <= 0) {
            throw new Error("Invalid data pointer");
        }
      
        return new Uint8ClampedArray(this.Module.HEAPU8.buffer, data_ptr, total_size);
    }

    destroy() {
        if (this.mosaic_ptr) {
            this.Module._deleteMosaic(this.mosaic_ptr);
            this.mosaic_ptr = null;
        } 
    }

  }
  