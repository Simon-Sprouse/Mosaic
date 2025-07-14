#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#include <opencv2/opencv.hpp>
#include <string>


namespace Graphics { 

    void drawSquare(cv::Mat& image, const cv::Point& center, double size, double angle_deg, const cv::Scalar& color, int border_width);
    void drawSquareText(cv::Mat& image, const cv::Point& center, double size, double angle_deg, const cv::Scalar& color, int border_width, const std::string& text);
    void drawLine(cv::Mat& image, const cv::Point& point_a, const cv::Point& point_b, int thickness, const cv::Scalar& color);
}

#endif