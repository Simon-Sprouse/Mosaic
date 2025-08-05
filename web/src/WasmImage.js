export default class WasmImage {

    constructor(Module) {
        this.Module = Module;
        this.image_ptr = null;
    }

    getMessage() { 
        const message_ptr = this.Module._helloWorld();
        const message_string = this.Module.UTF8ToString(message_ptr);
        return message_string;
    }


    loadImageFromBytes(byteArray) { 

        if (this.image_ptr) {
            this.Module._deleteImage(this.image_ptr); // ✅ free old image
            this.image_ptr = null;
        }
        
        const length = byteArray.length;
        const ptr = this.Module._malloc(length);
        this.Module.HEAPU8.set(byteArray, ptr);

        this.image_ptr = this.Module._loadImageFromBytes(ptr, length);  // assumes C++ function exists

        this.Module._free(ptr);
    }

    empty() {
        if (!this.image_ptr) return true;  // No image loaded yet = empty
        return !!this.Module._empty(this.image_ptr);  // Cast C++ bool (0/1) to JS bool
    }

    getSize() {
        const string_ptr = this.Module._getSizeStr(this.image_ptr);
        const size_string = this.Module.UTF8ToString(string_ptr);
        const [width, height] = size_string.split(',').map(Number);
        return { width, height };
    }

    getLength() { 
        const { width, height } = this.getSize();
        return width * height;
    }

    getDataArray() {

    
        if (!this.Module) {
            throw new Error("Module not available")
        }
        if (!this.Module.HEAPU8) {
            throw new Error("Module.HEAPU8 not available");
          }
      
        const data_ptr = this.Module._getData(this.image_ptr);
        const total_size = this.getLength() * 4; // assuming RGBA
      
        if (typeof data_ptr !== 'number' || data_ptr <= 0) {
            throw new Error("Invalid data pointer");
        }
      
        return new Uint8ClampedArray(this.Module.HEAPU8.buffer, data_ptr, total_size);
    }

    destroy() {
        if (this.image_ptr) {
            this.Module._deleteImage(this.image_ptr);
            this.image_ptr = null;
        } 
    }

  }
  