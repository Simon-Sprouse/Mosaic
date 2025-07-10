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


    void drawSquareText(cv::Mat& image, const cv::Point& center, double size, double angle_deg, const cv::Scalar& color, int border_width, const std::string& text) {
        // Draw the square
        drawSquare(image, center, size, angle_deg, color, border_width);

        if (text == "") { 
            return;
        }
    
        // Set font parameters
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
    
        // Get text size
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    
        // Compute top-left corner to center the text
        cv::Point textOrg(center.x - textSize.width / 2, center.y + textSize.height / 2);
    
        // Draw black rectangle behind text for legibility
        cv::Point rectTopLeft(textOrg.x, textOrg.y - textSize.height);
        cv::Point rectBottomRight(textOrg.x + textSize.width, textOrg.y + baseline);
        cv::rectangle(image, rectTopLeft, rectBottomRight, cv::Scalar(0, 0, 0), cv::FILLED);
    
        // Draw white text
        cv::putText(image, text, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
    
    

}