#include "../Image/Image.hpp"
#include <cstdint>


extern "C" {

    const char* helloWorld() {
        return "Hello world";
    }

    image::Image* defaultConstructImage() {
        return new image::Image(1, 1);
    }

    const char* getSize(image::Image* img) { 
        static std::string result = img->size().toString();
        return result.c_str();
    }

    void deleteImage(image::Image* img) { 
        delete img;
    }

    // image::Image* create_image(int width, int height) {
    //     return new image::Image(width, height);
    // }

    // void fill_blue(image::Image* img) {
    //     for (int y = 0; y < img->size().height; ++y) {
    //         for (int x = 0; x < img->size().width; ++x) {
    //             img->at(x, y) = image::Color(0, 0, 255); // Blue
    //         }
    //     }
    // }

    // uint8_t* get_image_data(image::Image* img) {
    //     return reinterpret_cast<uint8_t*>(img->rawData()); // implement rawData()
    // }

    // int get_image_size(image::Image* img) {
    //     return img->size().width * img->size().height * 4;
    // }
}
