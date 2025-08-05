# /Code/Utils/io/Makefile

EMCC = emcc
SRC = Interface.cpp ../Image/Image.cpp ../ImageProcess/imageProcess.cpp ../Random/random.cpp ../Geometry/geometry.cpp ../Graphics/graphics.cpp ../../Modules/Mosaic/Mosaic.cpp
OUT_DIR = ../../../web/public/wasm
OUT = $(OUT_DIR)/image_module.js

CXXFLAGS = -O3 -std=c++17
EXPORTS="['_helloWorld','_loadImageFromBytes','_getData','_getSizeStr','_empty','_deleteImage','_loadMosaicFromBytes','_getMosaicOutput','_getMosaicSizeStr','_mosaicIsEmpty','_deleteMosaic','_malloc','_free']"

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
