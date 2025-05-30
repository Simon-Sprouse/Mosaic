#ifndef IMAGE_PROCESS_HPP
#define IMAGE_PROCESS_HPP

#include <string>
#include <opencv2/opencv.hpp>

namespace ImageProcess { 
    
    struct ImageState { 
        cv::Mat original;
        cv::Mat rescaled;
        cv::Mat grayscale;
        cv::Mat edges;
        cv::Mat mask;
        std::string name;
        std::string file_path;
    };

    ImageState loadImage(const std::string& path);
    
    void saveImage(const cv::Mat& image, const std::string& output_dir, const std::string& output_name);

}

#endif