#ifndef IMAGE_PROCESS_HPP
#define IMAGE_PROCESS_HPP

#include <string>
#include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/imgcodecs.hpp>

namespace ImageProcess { 
    
    struct ImageState { 
        cv::Mat original;
        cv::Mat resized;
        cv::Mat grayscale;
        cv::Mat blurred;
        cv::Mat edges;
        cv::Mat segmented;
        cv::Mat mask;
        std::string file_name;
        std::string file_path;

        // param constructor
        ImageState(const std::string& image_path);
    };

    
    
    void resizeImage(ImageState& state, double resize_factor);
    void grayImage(ImageState& state);
    void blurImage(ImageState& state, int kernel_size, double sigma);
    void cannyFilter(ImageState& state, int threshold_1, int threshold_2);
    int detectContours(ImageState& state, double max_segment_angle, int min_segment_length, int segment_angle_window);
    
    void saveImage(const cv::Mat& image, const std::string& output_dir, const std::string& output_name, const std::string& suffix);

    

}

#endif