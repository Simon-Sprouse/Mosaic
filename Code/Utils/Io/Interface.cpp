#include "../Image/Image.hpp"
#include "../Random/random.hpp"
#include <cstdint>


extern "C" {

    const char* helloWorld() {
        return "Hello from Interface.cpp";
    }



    image::Image* loadImageFromBytes(uint8_t* data, size_t length) {
        return new image::Image(image::fromEncodedBuffer(data, length));
    }

    const char* getSizeStr(image::Image* img) { 
        static char buffer[32]; // big enough for "12345,12345\0"
        std::string result = img->size().toString();
        strncpy(buffer, result.c_str(), sizeof(buffer));
        buffer[sizeof(buffer) - 1] = '\0'; // null-terminate
        return buffer;
    }
    

    bool empty(image::Image* img) { 
        return img->empty();
    }

    uint8_t* getData(image::Image* img) {
        return img->rawData();
    }





    image::Image* constructImage(int w, int h) {
        image::Color color = Random::randomColor();
        return new image::Image(w, h, color);
    }

    

    

    void deleteImage(image::Image* img) { 
        delete img;
    }



    





















   
}
