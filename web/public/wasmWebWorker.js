// public/wasmWebWorker.js


// vvv Debug script vvv
// fetch('/wasm/image_module.wasm').then(async (res) => {
//   const text = await res.text();
//   console.log('WASM file fetch response (as text):', text.slice(0, 100));
// }).catch(console.error);


let wasmInstance = null;
let wasmIsReady = false; // if called but not ready -> error is thrown


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
    }



};
