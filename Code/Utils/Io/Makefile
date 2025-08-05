# /Code/Utils/io/Makefile

EMCC = emcc
SRC = Interface.cpp ../Image/Image.cpp ../Random/random.cpp
OUT_DIR = ../../../web/public/wasm
OUT = $(OUT_DIR)/image_module.js

CXXFLAGS = -O3 -std=c++17
EXPORTS="['_helloWorld','_loadImageFromBytes','_constructImage','_getData','_getSizeStr','_empty','_deleteImage','_malloc','_free']"

RUNTIME_METHODS="['UTF8ToString','HEAPU8']"

all:
	mkdir -p $(OUT_DIR)
	$(EMCC) $(SRC) -o $(OUT) \
		-sEXPORTED_FUNCTIONS=$(EXPORTS) \
		-sEXPORTED_RUNTIME_METHODS=$(RUNTIME_METHODS) \
		-sMODULARIZE=1 \
		-sEXPORT_NAME=Module \
		-sASSERTIONS \
		$(CXXFLAGS)

clean:
	rm -f $(OUT) $(OUT:.js=.wasm)
