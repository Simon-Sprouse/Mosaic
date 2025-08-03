export default class WasmImage {
    constructor(Module) {
        this.Module = Module;
        this.img_ptr = null;
    }

    getMessage() { 
        const message_ptr = this.Module._helloWorld();
        const message_string = this.Module.UTF8ToString(message_ptr);
        return message_string;
    }

    constructImage(width, height) {
        this.image_ptr = this.Module._constructImage(width, height);
        this.width = width;
        this.height = height;
    }


    loadImageFromBuffer(byteArray) { 
        const length = byteArray.length;
        const ptr = this.Module._malloc(length);
        this.Module.HEAPU8.set(byteArray, ptr);

        this.image_ptr = this.Module._loadImageFromBytes(ptr, length);  // assumes C++ function exists

        this.Module._free(ptr);
    }






  
    getSizeStr() {
      const strPtr = this.Module._getSize(this.image_ptr);
      return this.Module.UTF8ToString(strPtr);
    }

    getDataArray() {

    
        if (!this.Module) {
            throw new Error("Module not available")
        }
        if (!this.Module.HEAPU8) {
            throw new Error("Module.HEAPU8 not available");
          }
      
        const ptr = this.Module._getData(this.image_ptr);

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
  