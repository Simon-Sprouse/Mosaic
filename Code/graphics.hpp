#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#include <opencv2/opencv.hpp>


namespace Graphics { 

    void drawSquare(cv::Mat& image, const cv::Point& center, double size, double angle_deg, const cv::Scalar& color, int border_width);

}

#endif