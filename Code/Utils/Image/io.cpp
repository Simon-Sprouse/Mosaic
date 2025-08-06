#include "io.hpp"
#include "Image.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include <iostream>
#include <fstream>

using namespace std;


namespace image::io { 

   


    Image loadImageFileSystem(const std::string& path) { 
        std::ifstream f(path, std::ios::binary);
        auto data = std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        return fromEncodedBuffer(data.data(), data.size());
    }

    void saveImageFileSystem(const Image& img, const std::string& save_path) {
        int width = img.getWidth();
        int height = img.getHeight();
        int channels = 4; // RGBA
    
        // Allocate flat buffer (row-major RGB)
        std::vector<uint8_t> buffer(width * height * channels);
    
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Color c = img.at(x, y);
                int idx = (y * width + x) * channels;
                buffer[idx + 0] = c.r;
                buffer[idx + 1] = c.g;
                buffer[idx + 2] = c.b;
                buffer[idx + 3] = c.a;
            }
        }
    
        // Write as PNG (you can also use stbi_write_jpg or stbi_write_bmp)
        int success = stbi_write_jpg(save_path.c_str(), width, height, channels, buffer.data(), width * channels);
    
        if (!success) {
            throw std::runtime_error("Failed to save image to: " + save_path);
        }
    }
    


}