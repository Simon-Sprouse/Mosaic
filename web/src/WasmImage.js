export default class WasmImage {
    constructor(Module) {
      this.Module = Module;
      
    }

    constructImage(width, height) {
        this.ptr = this.Module._constructImage(width, height);
        this.width = width;
        this.height = height;
    }
  
    getSizeStr() {
      const strPtr = this.Module._getSize(this.ptr);
      return this.Module.UTF8ToString(strPtr);
    }

    getDataArray() {

        
        
        if (!this.Module) {
            throw new Error("Module not available")
        }
        if (!this.Module.HEAPU8) {
            throw new Error("Module.HEAPU8 not available");
          }
      
        const ptr = this.Module._getData(this.ptr);

        const totalSize = this.width * this.height * 4; // assuming RGB
      
        if (typeof ptr !== 'number' || ptr <= 0) {
          throw new Error("Invalid data pointer");
        }
      
        return new Uint8ClampedArray(this.Module.HEAPU8.buffer, ptr, totalSize);
      }
      


  
    destroy() {
      if (this.ptr) {
        this.Module._deleteImage(this.ptr);
        this.ptr = null;
      }
    }
  }
  