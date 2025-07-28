#include "random.hpp"



namespace Random { 





    image::Color getRandomColor() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dist(0, 255);
    
        return image::Color{
            static_cast<uint8_t>(dist(gen)),  // red
            static_cast<uint8_t>(dist(gen)),  // green
            static_cast<uint8_t>(dist(gen))   // blue
        };
    }


        
    double randomDouble(double min_val, double max_val) {
        static std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(min_val, max_val);
        return dist(rng);
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

    // std::vector<cv::Point> samplePointsRandom(const cv::Mat& image, int num_points) {
    //     std::vector<cv::Point> random_points;

    //     if (image.empty() || num_points <= 0) {
    //         return random_points;
    //     }

    //     int width = image.cols;
    //     int height = image.rows;

    //     static std::mt19937 rng(std::random_device{}());
    //     std::uniform_int_distribution<int> dist_x(0, width - 1);
    //     std::uniform_int_distribution<int> dist_y(0, height - 1);

    //     for (int i = 0; i < num_points; ++i) {
    //         int x = dist_x(rng);
    //         int y = dist_y(rng);
    //         random_points.emplace_back(x, y);
    //     }

    //     return random_points;
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