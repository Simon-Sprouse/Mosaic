#include "../Image/Image.hpp"
#include "../Random/random.hpp"
#include <cstdint>


extern "C" {

    const char* helloWorld() {
        return "Hello world";
    }

    image::Image* constructImage(int w, int h) {
        image::Color color = Random::getRandomColor();
        return new image::Image(w, h, color);
    }

    uint8_t* getData(image::Image* img) {
        return img->rawData();
    }

    const char* getSize(image::Image* img) { 
        static std::string result = img->size().toString();
        return result.c_str();
    }

    void deleteImage(image::Image* img) { 
        delete img;
    }



    





















   
}
