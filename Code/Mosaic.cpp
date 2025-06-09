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

}