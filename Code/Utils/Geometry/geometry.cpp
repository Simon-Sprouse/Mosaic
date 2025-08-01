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

    









    double euclideanDistance(const Point& a, const Point& b) {
        double dx = static_cast<double>(a.x - b.x);
        double dy = static_cast<double>(a.y - b.y);
        return std::sqrt(dx * dx + dy * dy);
    }

    double vectorToAngleDegrees(const Vec2d& vec) {
        return std::atan2(vec.y, vec.x) * 180.0 / CV_PI;
    }


    Vec2d pcaDirection(const std::vector<Point>& points) {
        if (points.size() < 2) {
            return Vec2d();
        }
    
        Point mean = computeMean(points);
        auto centered = centerData(points, mean);
        auto cov = computeCovarianceMatrix(centered);
        return computeFirstEigenvector(cov);
    }


    std::vector<Point> filterUniquePoints(const std::vector<Point>& points, double min_distance) { 

        std::vector<Point> unique_points;
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

    void sortStrokesPCALength(std::vector<std::vector<Point>>& strokes) { 

        std::vector<double> segment_lengths;
        
    
        for (const std::vector<Point>& segment_pixels : strokes) {
            double length = Geometry::pcaLength(segment_pixels);
            segment_lengths.push_back(length);
        }
    
        // Pair lengths with their corresponding segment
        std::vector<std::pair<double, std::vector<Point>>> paired;
    
        for (size_t i = 0; i < segment_lengths.size(); ++i) {
            paired.emplace_back(segment_lengths[i], strokes[i]);
        }
    
        // Sort by length (ascending)
        std::sort(paired.begin(), paired.end(),
            [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
    
        // Unpack back into segments and lengths
        for (size_t i = 0; i < paired.size(); ++i) {
            strokes[i] = std::move(paired[i].second);
        }
    
    
    }

}