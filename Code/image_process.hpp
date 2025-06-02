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
        cv::Mat edges;
        cv::Mat mask;
        std::string file_name;
        std::string file_path;

        // param constructor
        ImageState(const std::string& image_path);
    };

    
    
    void resizeImage(ImageState& state, double resize_factor);


    
    void saveImage(const cv::Mat& image, const std::string& output_dir, const std::string& output_name, const std::string& suffix);

    

}

#endif