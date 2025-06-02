#include "image_process.hpp"
#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = std::__fs::filesystem;

namespace ImageProcess { 


// IMAGE STATE CONSTRUCTOR
ImageState::ImageState(const string& image_path) { 

    original = cv::imread(image_path);

    if (original.empty()) { 
        cerr << "Error: Could not load image from path: " << image_path << endl;
        return;
    }

    file_path = image_path;
    file_name = fs::path(image_path).stem().string();


}


void resizeImage(ImageState& state, double resize_factor) { 

    // if state.rescaled already exists we need to replace it / handle old version.
    // if state.resalced hasn't been set yet, we need to initialize it

    // set state.rescaled to cv::Mat with resalce performed


    if (state.original.empty()) { 
        cerr << "Resized called but no original image found" << endl;
        return;
    }

    cv::resize(state.original, state.resized, cv::Size(), resize_factor, resize_factor, cv::INTER_LINEAR);

}





// USED FOR TESTING //
void saveImage(const cv::Mat& image, const string& output_dir, const string& output_name, const string& suffix) { 
    if (image.empty()) { 
        return;
    }

    if (!fs::exists(output_dir)) { 
        fs::create_directory(output_dir);
    }

    std::string output_path = output_dir + "/" + output_name + "_" + suffix + ".jpg";
    if (cv::imwrite(output_path, image)) { 
        // cout << "Saved: " << output_path << endl;
    }
    else { 
        cerr << "Failed to save: " << output_path << endl;
    }


}

}