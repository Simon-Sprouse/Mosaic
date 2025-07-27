#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace Geometry { 

    double euclideanDistance(const cv::Point& a, const cv::Point& b);
    double vectorToAngleDegrees(const cv::Vec2d& vec);
    double pcaLength (const std::vector<cv::Point>& points);
    cv::Vec2d pcaDirection(const std::vector<cv::Point2d>& points);
    std::vector<cv::Point> filterUniquePoints(const std::vector<cv::Point>& points, double min_distance);

}