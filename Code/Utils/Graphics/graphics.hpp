#pragma once

#include "../Image/Image.hpp"

#include <opencv2/opencv.hpp>
#include <string>


namespace Graphics { 

    using image::Image;
    using image::Point;
    using image::Color;


    void drawFilledPolygon(Image& image, const std::vector<Point>& polygon, const Color& color);
    void drawSquare(Image& image, const Point& center, double size, double angle_deg, const Color& color, int border_width);



    // vvv Old Code vvv

    // void drawSquare(cv::Mat& image, const cv::Point& center, double size, double angle_deg, const cv::Vec3b& color, int border_width);
    // void drawSquareText(cv::Mat& image, const cv::Point& center, double size, double angle_deg, const cv::Vec3b& color, int border_width, const std::string& text);
    // void drawLine(cv::Mat& image, const cv::Point& point_a, const cv::Point& point_b, int thickness, const cv::Vec3b& color);
    // void drawArrow(cv::Mat& image, const cv::Point& center, int length, int thickness, double angle_deg, const cv::Vec3b& color);
}

