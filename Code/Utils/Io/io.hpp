#pragma once

#include "Image.hpp"

#include <string>
#include <opencv2/opencv.hpp>


namespace io {

    image::Image loadImageFileSystem(const std::string& path);
    image::Image loadImageFromCv(const std::string& image_path);
    void saveImage(const image::Image& img, const std::string& path);
    image::Image fromCvMat(const cv::Mat& mat);
    cv::Mat toCvMat(const image::Image& img);

}