#include "graphics.hpp"
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

namespace Graphics { 

    void drawSquare(cv::Mat& image, const cv::Point& center, double size, double angle_deg, const cv::Vec3b& color, int border_width) {
        if (image.empty()) {
            std::cerr << "drawSquare: Input image is empty." << std::endl;
            return;
        }
    
        if (border_width < 0) {
            std::cerr << "drawSquare: Invalid border_width: " << border_width << std::endl;
            return;
        }
    
        double half_size = size / 2.0;
    
        // If border is too big, just draw the whole filled square
        if (border_width * 2 >= static_cast<int>(size)) {
            double theta = angle_deg * CV_PI / 180.0;
            std::vector<cv::Point2d> outerCorners = {
                {-half_size, -half_size},
                { half_size, -half_size},
                { half_size,  half_size},
                {-half_size,  half_size}
            };
            std::vector<cv::Point> rotatedCorners;
            for (const auto& pt : outerCorners) {
                float x = pt.x * std::cos(theta) - pt.y * std::sin(theta);
                float y = pt.x * std::sin(theta) + pt.y * std::cos(theta);
                rotatedCorners.emplace_back(cvRound(center.x + x), cvRound(center.y + y));
            }
            std::vector<std::vector<cv::Point>> contour{ rotatedCorners };
            cv::fillPoly(image, contour, color);
            return;
        }
    
        // Compute rotation matrix
        double theta = angle_deg * CV_PI / 180.0;
        auto rotate = [theta](cv::Point2d pt) -> cv::Point2d {
            return {
                pt.x * std::cos(theta) - pt.y * std::sin(theta),
                pt.x * std::sin(theta) + pt.y * std::cos(theta)
            };
        };
    
        double inner_half = half_size - border_width;
    
        // Local-space corners
        std::vector<cv::Point2d> outer = {
            {-half_size, -half_size},
            { half_size, -half_size},
            { half_size,  half_size},
            {-half_size,  half_size}
        };
        std::vector<cv::Point2d> inner = {
            {-inner_half, -inner_half},
            { inner_half, -inner_half},
            { inner_half,  inner_half},
            {-inner_half,  inner_half}
        };
    
        // Rotate and shift to world coordinates
        std::vector<cv::Point> outerPts, innerPts;
        for (int i = 0; i < 4; ++i) {
            auto o = rotate(outer[i]);
            auto io = rotate(inner[i]);
            outerPts.emplace_back(cvRound(center.x + o.x), cvRound(center.y + o.y));
            innerPts.emplace_back(cvRound(center.x + io.x), cvRound(center.y + io.y));
        }
    
        // Draw 4 trapezoids (border strips)
        for (int i = 0; i < 4; ++i) {
            int next = (i + 1) % 4;
            std::vector<cv::Point> quad = {
                outerPts[i],
                outerPts[next],
                innerPts[next],
                innerPts[i]
            };
            std::vector<std::vector<cv::Point>> contour{ quad };
            cv::fillPoly(image, contour, color);
        }
    }
    


    void drawSquareText(cv::Mat& image, const cv::Point& center, double size, double angle_deg, const cv::Vec3b& color, int border_width, const std::string& text) {
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
    


    void drawLine(cv::Mat& image, const cv::Point& point_a, const cv::Point& point_b, int thickness, const cv::Vec3b& color) {
        if (image.empty()) {
            std::cerr << "drawLine: Input image is empty." << std::endl;
            return;
        }
    
        cv::line(image, point_a, point_b, color, thickness, cv::LINE_AA);
    }


    void drawArrow(cv::Mat& image, const cv::Point& center, int length, int thickness, double angle_deg, const cv::Vec3b& color) {
        if (image.empty()) {
            std::cerr << "drawArrow: Input image is empty." << std::endl;
            return;
        }
    
        // Parameters for arrow size (in pixels)

        const int headLength = static_cast<int>(length / 3);        // length of each arrowhead wing
        const double headAngleDeg = 30;  // angle between shaft and arrowhead wing

    
        // Convert angle to radians
        double theta = angle_deg * CV_PI / 180.0;
    
        // Compute arrow tip point
        cv::Point tip(
            cvRound(center.x + length * cos(theta)),
            cvRound(center.y + length * sin(theta))
        );
    
        // Draw shaft line
        cv::line(image, center, tip, color, thickness, cv::LINE_AA);
    
        // Calculate left wing point
        double leftTheta = theta + (CV_PI * headAngleDeg / 180.0);
        cv::Point leftWing(
            cvRound(tip.x - headLength * cos(leftTheta)),
            cvRound(tip.y - headLength * sin(leftTheta))
        );
    
        // Calculate right wing point
        double rightTheta = theta - (CV_PI * headAngleDeg / 180.0);
        cv::Point rightWing(
            cvRound(tip.x - headLength * cos(rightTheta)),
            cvRound(tip.y - headLength * sin(rightTheta))
        );
    
        // Draw arrowhead wings
        cv::line(image, tip, leftWing, color, thickness, cv::LINE_AA);
        cv::line(image, tip, rightWing, color, thickness, cv::LINE_AA);
    }
    
    

}