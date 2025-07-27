#pragma once

#include <random>
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>

namespace Random { 

    template<typename T>
    T selectFromVector(const std::vector<T>& vec) { 
         // Random engine and distribution
        static std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<> dist(0, static_cast<int>(vec.size()) - 1);

        // Pick a random index and return the value
        return vec[dist(rng)];
    }

    template<typename T>
    void shuffleVector(std::vector<T>& vec) { 
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(vec.begin(), vec.end(), rng);
    }

    double randomDouble(double min_val, double max_val); 
    std::vector<cv::Point> samplePointsGrid(const cv::Mat& image, int grid_size);
    std::vector<cv::Point> samplePointsRandom(const cv::Mat& image, int num_points);
    std::vector<cv::Point> jitterPoints(const std::vector<cv::Point>& input_points, int max_step, const cv::Size& image_size);
}