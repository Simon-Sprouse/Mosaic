# /Code/Utils/Interface/Makefile
EMCC = emcc
SRC = Interface.cpp ../Image/Image.cpp ../ImageProcess/imageProcess.cpp ../Random/random.cpp ../Geometry/geometry.cpp ../Graphics/graphics.cpp ../../Modules/Mosaic/Mosaic.cpp
OUT_DIR = ../../../web/public/wasm
OUT = $(OUT_DIR)/image_module.js
CXXFLAGS = -O3 -std=c++17 -g

EXPORTS="['_malloc','_free']"
RUNTIME_METHODS="['UTF8ToString','HEAPU8','ccall','cwrap']"

all:
	mkdir -p $(OUT_DIR)
	$(EMCC) $(SRC) -o $(OUT) \
		--bind \
		-lembind \
		-sEXPORTED_FUNCTIONS=$(EXPORTS) \
		-sEXPORTED_RUNTIME_METHODS=$(RUNTIME_METHODS) \
		-sMODULARIZE=1 \
		-sEXPORT_NAME=Module \
		-sALLOW_MEMORY_GROWTH \
		-sINITIAL_MEMORY=67108864 \
		-sASSERTIONS \
		$(CXXFLAGS)

clean:
	rm -f $(OUT) $(OUT:.js=.wasm)