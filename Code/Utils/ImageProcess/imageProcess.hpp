#pragma once
#include "../Image/Image.hpp"

namespace image::process { 



    // nearest neighbor sampling - TODO more elegant sampling
    Size resize(Image& src, Image& dest, int w, int h);
    Size resize(Image& src, Image& dest, Size size);
    Size resize(Image& src, Image& dest, double ratio);

    void grayscale(const Image& src, Image& dest);

    void gaussianBlur(Image& src, Image& dest, Size kernel_size, double blur_sigma);
    void gaussianBlur(Image& src, Image& dest, int kernel_size, double blur_sigma);
    std::vector<double> generateGaussianKernel1D(int radius, double sigma);

    void sobelFilter(const Image& src, Image& dest_grad_x, Image& dest_grad_y);
    void visualizeSobel(const Image& src_grad_x, const Image& src_grad_y, Image& dest);

    void cannyFilter(Image& src, Image& dest, int canny_threshold_1, int canny_threshold_2);
    void sobelFilterRaw(const Image& src, std::vector<int>& gradX, std::vector<int>& gradY);
    
    void findContours(const Image& src_binary, std::vector<std::vector<Point>>& contours);
    int  divideIntoStrokes(const std::vector<std::vector<Point>>& cv_contours, 
        std::vector<std::vector<Point>>& segment_points, 
        Size image_size, 
        int segment_angle_window, 
        double max_segment_angle_rad, 
        int min_segment_length);


    // TODO separate logic for distance field and gradient of distance field
    std::vector<float> computeDistanceField(const Image& strokes_img_source);
    Image floatMapToGrayscaleImage(const std::vector<float>& data, Size size);
    void computeSobelGradients(const std::vector<float>& distance_map, Size size, std::vector<float>& grad_x, std::vector<float>& grad_y);
}