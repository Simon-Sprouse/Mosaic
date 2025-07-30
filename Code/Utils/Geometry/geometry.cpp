#include "geometry.hpp"
#include "Image.hpp"

#include <iostream>


using namespace std;

namespace Geometry { 

    using image::Point;
    using image::Vec2d;

   




    // Compute the mean of a set of 2D points
    Point computeMean(const std::vector<Point>& points) {
        double sumX = 0.0;
        double sumY = 0.0;
        for (const Point& p : points) {
            sumX += p.x;
            sumY += p.y;
        }
        double n = static_cast<double>(points.size());
        return { sumX / n, sumY / n };
    }

    // Center the data (subtract the mean from each point)
    std::vector<Point> centerData(const std::vector<Point>& points, const Point& mean) {
        std::vector<Point> centered;
        centered.reserve(points.size());
        for (const Point& p : points) {
            int new_x = p.x - mean.x;
            int new_y = p.y - mean.y;
            centered.push_back(Point(new_x, new_y));
        }
        return centered;
    }

    // Compute the 2x2 covariance matrix from centered points
    std::array<std::array<double, 2>, 2> computeCovarianceMatrix(const std::vector<Point>& points) {
        double sumXX = 0.0, sumXY = 0.0, sumYY = 0.0;
    
        for (const Point& p : points) {
            sumXX += p.x * p.x;
            sumXY += p.x * p.y;
            sumYY += p.y * p.y;
        }
    
        double n = static_cast<double>(points.size());
    
        return {{
            { sumXX / n, sumXY / n },
            { sumXY / n, sumYY / n }
        }};
    }
    

    Vec2d computeFirstEigenvector(const std::array<std::array<double, 2>, 2>& cov) {
        double a = cov[0][0];
        double b = cov[0][1];  // == cov[1][0]
        double d = cov[1][1];
    
    
        double theta = 0.5 * std::atan2(2 * b, a - d);
    
        double x = std::cos(theta);
        double y = std::sin(theta);

    
        return Vec2d(x, y);
    }
    

    
    

     


    // Computes PCA length along the first principal component
    double pcaLength(const std::vector<Point>& points) {
 

        if (points.size() < 2) {
            return 0.0;
        }

        // cout << "Points: " << endl;
        // for (const auto& p : points) {
        //     std::cout << "(" << p.x << ", " << p.y << ") ";
        // std::cout << "\n";
        // }

        Point mean = computeMean(points);
        std::vector<Point> centered = centerData(points, mean);

        
        auto cov = computeCovarianceMatrix(centered);

        Vec2d eigenVec = computeFirstEigenvector(cov);


        // Project points onto the eigenvector
        double minProj = std::numeric_limits<double>::max();
        double maxProj = std::numeric_limits<double>::lowest();

        for (const Point& p : centered) {
            double projection = p.x * eigenVec.x + p.y * eigenVec.y;
            if (projection < minProj) minProj = projection;
            if (projection > maxProj) maxProj = projection;
        }

        return maxProj - minProj;
    }

    







    // double pcaLength (const std::vector<image::Point>& points) {
    //     if (points.size() < 2)
    //         return 0.0;
    
    //     cv::Mat data(points.size(), 2, CV_64F);
    //     for (size_t i = 0; i < points.size(); ++i) {
    //         data.at<double>(i, 0) = points[i].x;
    //         data.at<double>(i, 1) = points[i].y;
    //     }
    
    //     cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
    //     cv::Mat projected;
    //     pca.project(data, projected);
    
    //     double minVal, maxVal;
    //     cv::minMaxLoc(projected.col(0), &minVal, &maxVal);
    
    //     return maxVal - minVal;
    // }


    double euclideanDistance(const cv::Point& a, const cv::Point& b) {
        double dx = static_cast<double>(a.x - b.x);
        double dy = static_cast<double>(a.y - b.y);
        return std::sqrt(dx * dx + dy * dy);
    }

    double vectorToAngleDegrees(const cv::Vec2d& vec) {
        return std::atan2(vec[1], vec[0]) * 180.0 / CV_PI;
    }


    cv::Vec2d pcaDirection(const std::vector<cv::Point>& points) { 
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