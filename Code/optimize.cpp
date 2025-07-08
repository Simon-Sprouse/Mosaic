#include "optimize.hpp"
#include <cmath>

using namespace std;

namespace Optimize { 

    double reward(cv::Mat segment_image, cv::Point center, double size, double theta, double decay_rate) { 

        int H = segment_image.rows;
        int W = segment_image.cols;

        double x0 = center.x;
        double y0 = center.y;

        double half = size / 2.0;
        double angle_rad = theta * CV_PI / 180.0;

        double reward = 0.0;

        int l_bound = -static_cast<int>(size) / 2;
        int r_bound = static_cast<int>(size) / 2;
        for (int dx = l_bound; dx <= r_bound; dx++) { 

            int dy = 0;

            // point on rotated center line
            double dy_rot = dy * cos(angle_rad) - dx * sin(angle_rad);
            double dx_rot = dy * sin(angle_rad) + dx * cos(angle_rad);

            int x = static_cast<int>(round(x0 + dx_rot));
            int y = static_cast<int>(round(y0 + dy_rot));

            if (x >= 0 && x < W && y >= 0 && y < H) { 
                reward += static_cast<double>(segment_image.at<uchar>(y, x));

            }

            //
            for (int offset = 1; offset <= 10; ++offset) {
                double weight = std::exp(-decay_rate * std::pow(offset / half, 2));
    
                // Above
                double dy_off = offset;
                dy_rot = dy_off * cos(angle_rad) - dx * sin(angle_rad);
                dx_rot = dy_off * sin(angle_rad) + dx * cos(angle_rad);
                int x_off = static_cast<int>(round(x0 + dx_rot));
                int y_off = static_cast<int>(round(y0 + dy_rot));
                if (x_off >= 0 && x_off < W && y_off >= 0 && y_off < H) {
                    reward += weight * static_cast<double>(segment_image.at<uchar>(y_off, x_off));
                }
    
                // Below
                dy_off = -offset;
                dy_rot = dy_off * cos(angle_rad) - dx * sin(angle_rad);
                dx_rot = dy_off * sin(angle_rad) + dx * cos(angle_rad);
                x_off = static_cast<int>(round(x0 + dx_rot));
                y_off = static_cast<int>(round(y0 + dy_rot));
                if (x_off >= 0 && x_off < W && y_off >= 0 && y_off < H) {
                    reward += weight * static_cast<double>(segment_image.at<uchar>(y_off, x_off));
                }
            }
    


        }



        return reward;
    }


    double rewardFromCanny(const cv::Mat& edge_image, cv::Point2f center, double size, double theta_deg, double decay_rate) {
        
        cv::Mat gray;
        if (edge_image.channels() > 1) {
            cv::cvtColor(edge_image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = edge_image;
        }

    std::vector<cv::Point> nonzero;
    cv::findNonZero(gray, nonzero);

    
        double half = size / 2.0;
        double theta = theta_deg * CV_PI / 180.0;
    
        // Direction vector along the centerline
        double vx = std::cos(theta);
        double vy = std::sin(theta);
    
        // Perpendicular vector to centerline
        double vpx = -vy;
        double vpy = vx;
    
        double reward = 0.0;
    
        for (const auto& pt : nonzero) {
            double dx = pt.x - center.x;
            double dy = pt.y - center.y;
    
            // Project onto perpendicular vector to get distance from centerline
            double dist = std::abs(dx * vpx + dy * vpy);
    
            // Check if within square bounds
            double along = std::abs(dx * vx + dy * vy);
            if (along <= half && dist <= half) {
                double weight = std::exp(-decay_rate * std::pow(dist / half, 2));
                reward += weight;
            }
        }
    
        return reward;
    }
    


    void find_best_theta() { 

    }
}