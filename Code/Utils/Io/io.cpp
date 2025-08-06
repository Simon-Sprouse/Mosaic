#include "io.hpp"
#include "Image.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include <iostream>
#include <fstream>

using namespace std;


namespace io { 

   


    image::Image loadImageFileSystem(const std::string& path) { 
        std::ifstream f(path, std::ios::binary);
        auto data = std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        return image::fromEncodedBuffer(data.data(), data.size());
    }

    void saveImageFileSystem(const image::Image& img, const std::string& save_path) {
        int width = img.getWidth();
        int height = img.getHeight();
        int channels = 3; // RGB
    
        // Allocate flat buffer (row-major RGB)
        std::vector<uint8_t> buffer(width * height * channels);
    
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                image::Color c = img.at(x, y);
                int idx = (y * width + x) * channels;
                buffer[idx + 0] = c.r;
                buffer[idx + 1] = c.g;
                buffer[idx + 2] = c.b;
            }
        }
    
        // Write as PNG (you can also use stbi_write_jpg or stbi_write_bmp)
        int success = stbi_write_jpg(save_path.c_str(), width, height, channels, buffer.data(), width * channels);
    
        if (!success) {
            throw std::runtime_error("Failed to save image to: " + save_path);
        }
    }
    


    image::Image loadImageFromCv(const std::string& image_path) { 
        
        cv::Mat mat = cv::imread(image_path);
    
        return fromCvMat(mat);

    }

    void saveImage(const image::Image& img, const std::string& path) {

        if (img.empty()) { 
            return;
        }

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