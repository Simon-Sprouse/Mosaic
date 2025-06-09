#include "Mosaic.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::__fs::filesystem;

namespace mosaic_gen {

// param constructor
Mosaic::Mosaic(const std::string& image_path) { 
    original = cv::imread(image_path);

    if (original.empty()) { 
        cerr << "Error: Could not load image from path: " << image_path << endl;
        return;
    }

    file_path = image_path;
    image_name = fs::path(image_path).stem().string();

}


void Mosaic::resizeOriginal(double resize_factor) { 
    if (original.empty()) { 
        cerr << "Resized called but no original image found" << endl;
        return;
    }

    cv::resize(original, resized, cv::Size(), resize_factor, resize_factor, cv::INTER_LINEAR);
}

void Mosaic::saveImage(const cv::Mat& image, const std::string& output_dir, const std::string& suffix) { 

    if (image.empty()) { 
        return;
    }

    if (!fs::exists(output_dir)) { 
        fs::create_directory(output_dir);
    }

    std::string output_path = output_dir + "/" + image_name + "_" + suffix + ".jpg";
    if (cv::imwrite(output_path, image)) { 
        // cout << "Saved: " << output_path << endl;
    }
    else { 
        cerr << "Failed to save: " << output_path << endl;
    }


}












}