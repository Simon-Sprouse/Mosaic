# /Code/Utils/io/Makefile

EMCC = emcc
SRC = Interface.cpp ../Image/Image.cpp
OUT_DIR = ../../../web/public/wasm
OUT = $(OUT_DIR)/image_module.js

CXXFLAGS = -O3 -std=c++17
EXPORTS="['_helloWorld','_defaultConstructImage','_getSize','_deleteImage']"

RUNTIME_METHODS = ['UTF8ToString']

all:
	mkdir -p $(OUT_DIR)
	$(EMCC) $(SRC) -o $(OUT) \
		-sEXPORTED_FUNCTIONS=$(EXPORTS) \
		-sEXPORTED_RUNTIME_METHODS=$(RUNTIME_METHODS) \
		-sMODULARIZE=1 \
		-sEXPORT_NAME=Module \
		$(CXXFLAGS)

clean:
	rm -f $(OUT) $(OUT:.js=.wasm)
