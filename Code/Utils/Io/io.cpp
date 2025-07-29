#include "io.hpp"

#include <iostream>

using namespace std;


namespace io { 


    image::Image loadImage(const std::string& image_path) { 
        
        cv::Mat mat = cv::imread(image_path);
    
        return fromCvMat(mat);

    }

    void saveImage(const image::Image& img, const std::string& path) {
        cv::Mat mat = toCvMat(img);
        if (!cv::imwrite(path, mat)) {
            throw std::runtime_error("Failed to save image to: " + path);
        }
    
    }
    


    image::Image fromCvMat(const cv::Mat& mat) {
        CV_Assert(mat.type() == CV_8UC3);  // Make sure it's 3-channel uchar (RGB)

        image::Image img(mat.cols, mat.rows);

        for (int y = 0; y < mat.rows; ++y) {
            for (int x = 0; x < mat.cols; ++x) {
                const cv::Vec3b& pixel = mat.at<cv::Vec3b>(y, x);
                image::Color color(pixel[2], pixel[1], pixel[0]); // Convert BGR → RGB
                img.setPixel(x, y, color);
            }
        }

        return img;
    }

    cv::Mat toCvMat(const image::Image& img) {
        cv::Mat mat(img.getHeight(), img.getWidth(), CV_8UC3);  // OpenCV stores as rows x cols
    
        for (int y = 0; y < img.getHeight(); ++y) {
            for (int x = 0; x < img.getWidth(); ++x) {
                image::Color color = img.at(x, y);
                // OpenCV stores in BGR order
                mat.at<cv::Vec3b>(y, x) = cv::Vec3b(color.b, color.g, color.r);
            }
        }
    
        return mat;
    }
    


}