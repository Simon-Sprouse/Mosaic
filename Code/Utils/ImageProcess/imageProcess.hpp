#pragma once
#include "Image.hpp"

namespace image::process { 

    Size resize(Image& src, Image& dest, int w, int h);
    Size resize(Image& src, Image& dest, Size size);
    Size resize(Image& src, Image& dest, double ratio);

    void grayscale(Image& src, Image& dest);

}