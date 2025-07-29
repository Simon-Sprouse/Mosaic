#include "imageProcess.hpp"

namespace image::process { 




    Size resize(Image& src, Image& dest, int w, int h) {
        if (w <= 0 || h <= 0) {
            throw std::invalid_argument("resize: width and height must be positive");
        }

        Size newSize(w, h);
        dest = Image(newSize);  // Reallocate destination image with new size


        const double scaleX = static_cast<double>(src.getWidth()) / w;
        const double scaleY = static_cast<double>(src.getHeight()) / h;

        for (int y = 0; y < h; ++y) {
            int srcY = static_cast<int>(y * scaleY);
            if (srcY >= src.getHeight()) srcY = src.getHeight() - 1;

            for (int x = 0; x < w; ++x) {
                int srcX = static_cast<int>(x * scaleX);
                if (srcX >= src.getWidth()) srcX = src.getWidth() - 1;

                dest.setPixel(x, y, src.at(srcX, srcY));
            }
        }

        return newSize;
    }

    Size resize(Image& src, Image& dest, Size size) { 
        return resize(src, dest, size.height, size.width);
    }

    Size resize(Image& src, Image& dest, double ratio) { 
        int new_width = static_cast<int>(src.getWidth() * ratio);
        int new_height = static_cast<int>(src.getHeight() * ratio);

        return resize(src, dest, new_width, new_height);
    }



}