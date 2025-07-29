#pragma once
#include "Image.hpp"

namespace image::process { 

    // nearest neighbor sampling - TODO more elegant sampling
    Size resize(Image& src, Image& dest, int w, int h);
    Size resize(Image& src, Image& dest, Size size);
    Size resize(Image& src, Image& dest, double ratio);

    void grayscale(Image& src, Image& dest);

    void gaussianBlur(Image& src, Image& dest, Size kernel_size, double blur_sigma);
    std::vector<double> generateGaussianKernel1D(int radius, double sigma);

    void sobelFilter(const Image& src, Image& dest_grad_x, Image& dest_grad_y);
    void visualizeSobel(const Image& src_grad_x, const Image& src_grad_y, Image& dest);

    void cannyFilter(Image& src, Image& dest, int canny_threshold_1, int canny_threshold_2);
    void sobelFilterRaw(const Image& src, std::vector<int>& gradX, std::vector<int>& gradY);
    void linkEdge(int x, int y, int width, int height, std::vector<uint8_t>& edgeMap);


}