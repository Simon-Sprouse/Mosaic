export default class WasmImage {
    constructor(Module) {
      this.Module = Module;
      this.ptr = Module._defaultConstructImage();
    }
  
    getSizeStr() {
      const strPtr = this.Module._getSize(this.ptr);
      return this.Module.UTF8ToString(strPtr);
    }
  
    destroy() {
      if (this.ptr) {
        this.Module._deleteImage(this.ptr);
        this.ptr = null;
      }
    }
  }
  