#include "graphics.hpp"
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

namespace Graphics { 

    void drawSquare(cv::Mat& image, const cv::Point& center, double size, double angle_deg, const cv::Scalar& color, int border_width) {
        if (border_width <= 0) {
            std::cerr << "Border width invalid: " << border_width << std::endl;
            return;
        }
    
        float half_size = static_cast<float>(size / 2.0);
        double theta = angle_deg * M_PI / 180.0;
    
        std::vector<cv::Point2f> corners = {
            {-half_size, -half_size},
            {half_size, -half_size},
            {half_size, half_size},
            {-half_size, half_size}
        };
    
        std::vector<cv::Point> rotated_corners;
        for (const auto& point : corners) {
            double x_rot = point.x * cos(theta) - point.y * sin(theta);
            double y_rot = point.x * sin(theta) + point.y * cos(theta);
            rotated_corners.emplace_back(cv::Point(cvRound(center.x + x_rot), cvRound(center.y + y_rot)));
        }
    
        std::vector<std::vector<cv::Point>> outer_contour = { rotated_corners };
        cv::drawContours(image, outer_contour, 0, color, border_width, cv::LINE_AA);
    }
    

}