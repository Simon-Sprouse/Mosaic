#include "image_process.hpp"
#include <iostream>
#include <filesystem>
#include <random>
#include <cmath>

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

    if (state.original.empty()) { 
        cerr << "Resized called but no original image found" << endl;
        return;
    }

    cv::resize(state.original, state.resized, cv::Size(), resize_factor, resize_factor, cv::INTER_LINEAR);

}

void grayImage(ImageState& state) { 
    
    if (state.resized.empty()) { 
        cerr << "Gray called but no resized image" << endl;
        return;
    }
    cv::cvtColor(state.resized, state.grayscale, cv::COLOR_BGR2GRAY);

}

void blurImage(ImageState& state, int kernel_size, double sigma) { 
    // apply blur to state.grayscale and save in state.blurred
    if (state.grayscale.empty()) { 
        cerr << "Blur called but no grayscale image" << endl;
        return;
    }

    // ensure odd kernel size
    if (kernel_size % 2 == 0) { 
        kernel_size += 1;
    }

    cv::GaussianBlur(state.grayscale, state.blurred, cv::Size(kernel_size, kernel_size), sigma);

}

void cannyFilter(ImageState& state, int threshold_1, int threshold_2) {
    if (state.blurred.empty()) { 
        cerr << "Canny called but no blurred" << endl;
        return;
    }
    cv::Canny(state.blurred, state.edges, threshold_1, threshold_2);
}


int detectContours(ImageState& state, double max_segment_angle_rad, int min_segment_length, int segment_angle_window) {
    if (state.edges.empty()) {
        cerr << "DetectContours called but no edges" << endl;
        return -1;
    }

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(state.edges.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Create an output color image
    state.segmented = cv::Mat::zeros(state.edges.size(), CV_8UC3);
    int contour_id = 0;
    std::vector<cv::Vec3b> colors_used;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> color_dist(64, 255);

    for (const auto& contour : contours) {
        if (contour.size() < 3)
            continue;

        std::vector<int> breaks;
        int len = contour.size();
        int w = segment_angle_window;

        for (int i = w; i < len - w; ++i) {
            cv::Point2f v1 = contour[i] - contour[i - w];
            cv::Point2f v2 = contour[i + w] - contour[i];

            double norm1 = std::sqrt(v1.x * v1.x + v1.y * v1.y) + 1e-8;
            double norm2 = std::sqrt(v2.x * v2.x + v2.y * v2.y) + 1e-8;

            cv::Point2f n1 = v1 / norm1;
            cv::Point2f n2 = v2 / norm2;

            double cosine = std::clamp(n1.dot(n2), -1.0f, 1.0f);
            double angle = std::acos(std::abs(cosine));

            if (angle > max_segment_angle_rad) {
                breaks.push_back(i);
            }
        }

        // Build split indices
        std::vector<int> split_idxs = {0};
        split_idxs.insert(split_idxs.end(), breaks.begin(), breaks.end());
        split_idxs.push_back(len);

        for (size_t i = 0; i < split_idxs.size() - 1; ++i) {
            int a = split_idxs[i];
            int b = split_idxs[i + 1];
            if (b - a < min_segment_length)
                continue;

            // Generate a new color not used yet
            cv::Vec3b color;
            do {
                color = cv::Vec3b(color_dist(rng), color_dist(rng), color_dist(rng));
            } while (std::find(colors_used.begin(), colors_used.end(), color) != colors_used.end());

            colors_used.push_back(color);

            for (int j = a; j < b; ++j) {
                const auto& pt = contour[j];
                if (pt.y >= 0 && pt.y < state.segmented.rows && pt.x >= 0 && pt.x < state.segmented.cols) {
                    state.segmented.at<cv::Vec3b>(pt.y, pt.x) = color;
                }
            }

            ++contour_id;
        }
    }

    return contour_id;
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