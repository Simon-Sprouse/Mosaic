#pragma once
#include "Image.hpp"

namespace image::process { 
    Size resize(Image& src, Image& dest, int w, int h);
}