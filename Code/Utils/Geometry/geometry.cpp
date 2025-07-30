#include "geometry.hpp"


namespace Geometry { 

    double euclideanDistance(const cv::Point& a, const cv::Point& b) {
        double dx = static_cast<double>(a.x - b.x);
        double dy = static_cast<double>(a.y - b.y);
        return std::sqrt(dx * dx + dy * dy);
    }

    double vectorToAngleDegrees(const cv::Vec2d& vec) {
        return std::atan2(vec[1], vec[0]) * 180.0 / CV_PI;
    }

    double pcaLength (const std::vector<image::Point>& points) {
        if (points.size() < 2)
            return 0.0;
    
        cv::Mat data(points.size(), 2, CV_64F);
        for (size_t i = 0; i < points.size(); ++i) {
            data.at<double>(i, 0) = points[i].x;
            data.at<double>(i, 1) = points[i].y;
        }
    
        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
        cv::Mat projected;
        pca.project(data, projected);
    
        double minVal, maxVal;
        cv::minMaxLoc(projected.col(0), &minVal, &maxVal);
    
        return maxVal - minVal;
    }

    cv::Vec2d pcaDirection(const std::vector<cv::Point2d>& points) { 
        if (points.size() < 2) {
            return cv::Vec2d();
        }

        // Build matrix for PCA
        cv::Mat data(points.size(), 2, CV_64F);
        for (size_t i = 0; i < points.size(); ++i) {
            data.at<double>(i, 0) = points[i].x;
            data.at<double>(i, 1) = points[i].y;
        }

        // Run PCA to get dominant direction
        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
        cv::Vec2d direction = pca.eigenvectors.row(0);

        return direction;
    }


    std::vector<cv::Point> filterUniquePoints(const std::vector<cv::Point>& points, double min_distance) { 

        std::vector<cv::Point> unique_points;
        for (const auto& pt : points) {
            bool isFarEnough = true;
            for (const auto& kept : unique_points) {
                if (euclideanDistance(pt, kept) < min_distance) {
                    isFarEnough = false;
                    break;
                }
            }
            if (isFarEnough) {
                unique_points.push_back(pt);
            }
        }
        return unique_points;
    }

}