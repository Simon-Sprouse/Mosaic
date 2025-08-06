#pragma once

#include "Image.hpp"

#include <string>


// TODO make this part of image namespace
namespace image::io {

    image::Image loadImageFileSystem(const std::string& path);
    void saveImageFileSystem(const image::Image& img, const std::string& save_path);


}