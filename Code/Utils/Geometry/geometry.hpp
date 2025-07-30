#pragma once

#include "../Image/Image.hpp"

#include <vector>
#include <array>
#include <opencv2/core.hpp>


namespace Geometry { 

    using image::Point;
    using image::Vec2d;

   
    Point computeMean(const std::vector<Point>& points);
    std::vector<Point> centerData(const std::vector<Point>& points, const Point& mean);
    std::array<std::array<double, 2>, 2> computeCovarianceMatrix(const std::vector<Point>& points);
    Vec2d computeFirstEigenvector(const std::array<std::array<double, 2>, 2>& cov);
    double pcaLength(const std::vector<Point>& points);




    double euclideanDistance(const cv::Point& a, const cv::Point& b);
    double vectorToAngleDegrees(const cv::Vec2d& vec);
    cv::Vec2d pcaDirection(const std::vector<cv::Point2d>& points);
    std::vector<cv::Point> filterUniquePoints(const std::vector<cv::Point>& points, double min_distance);

}