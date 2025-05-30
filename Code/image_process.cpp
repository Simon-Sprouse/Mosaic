#include "image_process.hpp"
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::__fs::filesystem;

namespace ImageProcess { 

ImageState loadImage(const string& path) { 
    ImageState imgState;
    imgState.original = cv::imread(path);

    if (imgState.original.empty()) { 
        cerr << "Error: Could not load image from path: " << path << endl;
        return imgState;
    }

    imgState.file_path = path;
    imgState.name = fs::path(path).stem().string();

    return imgState;

}


void saveImage(const cv::Mat& image, const string& output_dir, const string& output_name) { 
    if (image.empty()) { 
        return;
    }

    if (!fs::exists(output_dir)) { 
        fs::create_directory(output_dir);
    }

    std::string output_path = output_dir + "/" + output_name + ".jpg";
    if (cv::imwrite(output_path, image)) { 
        cout << "Saved: " << output_path << endl;
    }
    else { 
        cout << "Failed to save: " << output_path << endl;
    }


}

}