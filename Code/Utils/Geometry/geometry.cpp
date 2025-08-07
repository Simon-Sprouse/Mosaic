#include "geometry.hpp"


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
        return std::atan2(vec.y, vec.x) * 180.0 / M_PI;
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




    double norm(const Vec2d& v) {
        return std::sqrt(v.x * v.x + v.y * v.y);
    }
    


    double pointLineSegmentDistance(const Vec2d& p, const Vec2d& A, const Vec2d& B) {
        Vec2d AB = B - A;
        Vec2d AP = p - A;
    
        double ab2 = AB.dot(AB);
        if (ab2 == 0.0) return norm(AP); // A == B case
    
        double t = AP.dot(AB) / ab2;
        t = std::max(0.0, std::min(1.0, t)); // Clamp t to [0,1]
    
        Vec2d projection = A + AB * t;
        return norm(p - projection);
    }









    std::vector<Point> samplePointsSquareBorder(Point center, double theta_deg, double distance_from_center, int num_points) {
        std::vector<Point> flood_points;
    
        // Convert center to float
        Vec2d center_f(center.x, center.y);
    
        // Convert angle to radians
        double theta_rad = theta_deg * M_PI / 180.0;
    
        // Rotated basis vectors
        Vec2d dx(std::cos(theta_rad), std::sin(theta_rad));       // along width
        Vec2d dy(-std::sin(theta_rad), std::cos(theta_rad));      // along height
    
        // Define half-size
        double h = distance_from_center;
    
        // Total perimeter of the square
        double perimeter = 8 * h;
    
        // Starting offset to reach middle of the top edge
        double start_offset = h;
    
        for (int i = 0; i < num_points; ++i) {
            double t = std::fmod(start_offset + (i * perimeter / num_points), perimeter);
    
            Vec2d local;
    
            if (t < 2 * h) {
                // Top edge (left to right)
                local = Vec2d(-h + t, -h);
            } else if (t < 4 * h) {
                // Right edge (top to bottom)
                local = Vec2d(h, -h + (t - 2 * h));
            } else if (t < 6 * h) {
                // Bottom edge (right to left)
                local = Vec2d(h - (t - 4 * h), h);
            } else {
                // Left edge (bottom to top)
                local = Vec2d(-h, h - (t - 6 * h));
            }
    
            // Rotate the local point using the dx/dy basis
            Vec2d rotated = center_f + dx * local.x + dy * local.y;
    
            // Convert to integer and store
            flood_points.push_back(Point(std::round(rotated.x), std::round(rotated.y)));
        }
    
        return flood_points;
    }
    












    std::vector<Point> getPointsInsideSquare(Point center, int size, double angle_deg) {
        std::vector<Point> result;


    
        double half = static_cast<double>(size) / 2.0;
        double angle_rad = angle_deg * M_PI / 180.0;
        double cos_a = std::cos(angle_rad);
        double sin_a = std::sin(angle_rad);
    
        // Conservative bounding box
        int min_x = static_cast<int>(std::floor(center.x - half - 1));
        int max_x = static_cast<int>(std::ceil(center.x + half + 1));
        int min_y = static_cast<int>(std::floor(center.y - half - 1));
        int max_y = static_cast<int>(std::ceil(center.y + half + 1));
    
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                // Translate to square center
                double dx = static_cast<double>(x - center.x);
                double dy = static_cast<double>(y - center.y);
    
                // Apply reverse rotation
                double rx =  dx * cos_a + dy * sin_a;
                double ry = -dx * sin_a + dy * cos_a;
    
                if (std::abs(rx) <= half && std::abs(ry) <= half) {
                    result.push_back(Point(x, y));
                }
            }
        }
    
        return result;
    }



}