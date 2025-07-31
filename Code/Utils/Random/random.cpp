#include "random.hpp"



namespace Random { 





    


        
    double randomDouble(double min_val, double max_val) {
        static std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(min_val, max_val);
        return dist(rng);
    }

    image::Color randomColor() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dist(0, 255);
    
        return image::Color{
            static_cast<uint8_t>(dist(gen)),  // red
            static_cast<uint8_t>(dist(gen)),  // green
            static_cast<uint8_t>(dist(gen))   // blue
        };
    }

    



    image::Point randomPoint(int w, int h) { 
        static std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist_x(0, w - 1);
        std::uniform_int_distribution<int> dist_y(0, h - 1);
        int x = dist_x(rng);
        int y = dist_y(rng);
        return image::Point(x, y);
    }
    image::Point randomPoint(image::Size size) {
        return randomPoint(size.width, size.height);
    }

    std::vector<image::Point> randomPointsVector(int w, int h, int num_points) { 

        std::vector<image::Point> random_points;

        static std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist_x(0, w - 1);
        std::uniform_int_distribution<int> dist_y(0, h - 1);

        for (int i = 0; i < num_points; ++i) {
            int x = dist_x(rng);
            int y = dist_y(rng);
            random_points.emplace_back(x, y);
        }

        return random_points;
    }
    std::vector<image::Point> randomPointsVector(image::Size size, int num_points) { 
        return randomPointsVector(size.width, size.height, num_points);
    }














    // std::vector<cv::Point> samplePointsGrid(const cv::Mat& image, int grid_size) { 
    //     std::vector<cv::Point> grid_points;

    //     if (image.empty() || grid_size <= 0) {
    //         return grid_points;
    //     }

    //     for (int y = 0; y < image.rows; y += grid_size) {
    //         for (int x = 0; x < image.cols; x += grid_size) {
    //             grid_points.emplace_back(x, y);
    //         }
    //     }

    //     return grid_points;
    // }




    // std::vector<cv::Point> jitterPoints(const std::vector<cv::Point>& input_points, int max_step, const cv::Size& image_size) {
    //     std::vector<cv::Point> jittered_points;

    //     if (max_step < 0 || input_points.empty()) {
    //         return input_points;  // No jittering needed
    //     }

    //     static std::mt19937 rng(std::random_device{}());
    //     std::uniform_int_distribution<int> offset_dist(-max_step, max_step);

    //     for (const auto& pt : input_points) {
    //         int jitter_x = pt.x + offset_dist(rng);
    //         int jitter_y = pt.y + offset_dist(rng);

    //         // Clamp to image bounds
    //         jitter_x = std::clamp(jitter_x, 0, image_size.width - 1);
    //         jitter_y = std::clamp(jitter_y, 0, image_size.height - 1);

    //         jittered_points.emplace_back(jitter_x, jitter_y);
    //     }

    //     return jittered_points;
    // }





}