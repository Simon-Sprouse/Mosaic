#pragma once

#include <random>
#include <stdexcept>
#include <vector>
// #include <opencv2/core.hpp>
#include "../Image/Image.hpp"

namespace Random { 

    using image::Point;
    using image::Size;
    using image::Color;



    double randomDouble(double min_val, double max_val); // TODO template this
    Color randomColor();

    Point randomPoint(int w, int h);
    Point randomPoint(Size size);
    std::vector<Point> randomPointsVector(int w, int h, int num_points);
    std::vector<Point> randomPointsVector(Size size, int num_points);


    std::vector<Point> gridPointsVector(int w, int h, int grid_size);
    std::vector<Point> gridPointsVector(Size size, int grid_size);
    std::vector<Point> jitterPoints(const std::vector<Point>& input_points, int max_step, const Size& image_size);



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

   

}