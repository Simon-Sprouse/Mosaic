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

void rankSegments(ImageState& state) {
    if (state.segmented.empty()) {
        std::cerr << "rankSegments called but segmented image is empty" << std::endl;
        return;
    }

    state.segment_pixels.clear();
    state.segment_lengths.clear();

    // Collect pixels for each color (excluding black)
    for (int y = 0; y < state.segmented.rows; ++y) {
        for (int x = 0; x < state.segmented.cols; ++x) {
            cv::Vec3b color = state.segmented.at<cv::Vec3b>(y, x);
            if (color != cv::Vec3b(0, 0, 0)) {
                state.segment_pixels[color].emplace_back(x, y);
            }
        }
    }

    // Helper lambda for PCA length
    auto pca_length = [](const std::vector<cv::Point>& points) -> double {
        if (points.size() < 2)
            return 0.0;

        cv::Mat data(points.size(), 2, CV_64F);
        for (size_t i = 0; i < points.size(); ++i) {
            data.at<double>(i, 0) = points[i].x;
            data.at<double>(i, 1) = points[i].y;
        }

        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
        cv::Mat projected;
        pca.project(data, projected);

        double minVal, maxVal;
        cv::minMaxLoc(projected.col(0), &minVal, &maxVal);

        return maxVal - minVal;
    };

    // Compute PCA length per color segment
    for (const auto& [color, pixels] : state.segment_pixels) {
        double length = pca_length(pixels);
        state.segment_lengths.emplace_back(color, length);
    }

    // Sort descending by length
    std::sort(state.segment_lengths.begin(), state.segment_lengths.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
}


void selectSegment(ImageState& state, int k) {
    if (state.segment_lengths.empty()) {
        std::cerr << "selectSegment called but segment_lengths is empty." << std::endl;
        return;
    }

    if (k < 0 || k >= static_cast<int>(state.segment_lengths.size())) {
        std::cerr << "selectSegment: k = " << k << " is out of range. Valid range: [0, "
                  << state.segment_lengths.size() - 1 << "]\n";
        return;
    }

    const cv::Vec3b& selected_color = state.segment_lengths[k].first;

    auto it = state.segment_pixels.find(selected_color);
    if (it == state.segment_pixels.end()) {
        std::cerr << "selectSegment: Selected color not found in segment_pixels.\n";
        return;
    }

    // Create a blank image
    state.selected_segment = cv::Mat::zeros(state.segmented.size(), CV_8UC3);

    // Draw only the selected segment
    for (const auto& pt : it->second) {
        if (pt.y >= 0 && pt.y < state.selected_segment.rows && pt.x >= 0 && pt.x < state.selected_segment.cols) {
            state.selected_segment.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(255, 255, 255);
        }
    }
}


cv::Point getRandomPointOnSegment(ImageState& state, int k) {
    // Safety check: make sure k is in bounds
    if (k < 0 || k >= static_cast<int>(state.segment_lengths.size())) {
        throw std::out_of_range("Segment index k is out of range");
    }

    // Get the color for segment k (assuming segment_lengths[k].first is the color)
    const cv::Vec3b& color = state.segment_lengths[k].first;

    // Find the vector of points corresponding to this color
    auto it = state.segment_pixels.find(color);
    if (it == state.segment_pixels.end()) {
        throw std::runtime_error("Color not found in segment_pixels");
    }

    const std::vector<cv::Point>& points = it->second;

    if (points.empty()) {
        throw std::runtime_error("No points in the selected segment");
    }

    // Random engine and distribution
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, static_cast<int>(points.size()) - 1);

    // Pick a random index and return the point
    return points[dist(rng)];
}





/*
PRINT FUNCTIONS >>
*/

// Helper to print Vec3b as (B,G,R)
std::string vec3bToString(const cv::Vec3b& color) {
    return "(" + std::to_string(color[0]) + ", " + 
                 std::to_string(color[1]) + ", " + 
                 std::to_string(color[2]) + ")";
}

// Helper to print Point as (x,y)
std::string pointToString(const cv::Point& pt) {
    return "(" + std::to_string(pt.x) + ", " + std::to_string(pt.y) + ")";
}

// Print color_to_pixels unordered_map
void printColorToPixels(const std::unordered_map<cv::Vec3b, std::vector<cv::Point>, ImageProcess::Vec3bHash, ImageProcess::Vec3bEqual>& color_to_pixels) {
    std::cout << "Color to Pixels Map:\n";
    for (const auto& [color, points] : color_to_pixels) {
        std::cout << "  Color " << vec3bToString(color) << " -> [";
        for (size_t i = 0; i < std::min(points.size(), size_t(5)); ++i) {
            std::cout << pointToString(points[i]);
            if (i != std::min(points.size(), size_t(5)) - 1) std::cout << ", ";
        }
        if (points.size() > 5) std::cout << "...";
        std::cout << "] (" << points.size() << " points)\n";
    }
}


// Print color_lengths vector
void printColorLengths(const std::vector<std::pair<cv::Vec3b, double>>& color_lengths) {
    std::cout << "Color Lengths:\n";
    for (const auto& [color, length] : color_lengths) {
        std::cout << "  Color " << vec3bToString(color) << " -> Length: " << length << "\n";
    }
}


void printColorToPixelsK(const std::unordered_map<cv::Vec3b, std::vector<cv::Point>, ImageProcess::Vec3bHash, ImageProcess::Vec3bEqual>& color_to_pixels) {
    int count = 0;
    int k = 5;
    std::cout << "Color to Pixels Map:\n";
    for (const auto& [color, points] : color_to_pixels) {
        if (count >= k) { 
            break;
        }
        std::cout << "  Color " << vec3bToString(color) << " -> [";
        for (size_t i = 0; i < std::min(points.size(), size_t(5)); ++i) {
            std::cout << pointToString(points[i]);
            if (i != std::min(points.size(), size_t(5)) - 1) std::cout << ", ";
        }
        if (points.size() > 5) std::cout << "...";
        std::cout << "] (" << points.size() << " points)\n";
        count++;
    }
}


void printColorLengthsK(const std::vector<std::pair<cv::Vec3b, double>>& color_lengths) {
    int count = 0;
    int k = 5;
    std::cout << "Color Lengths:\n";
    for (const auto& [color, length] : color_lengths) {
        if (count >= k) { 
            break;
        }
        std::cout << "  Color " << vec3bToString(color) << " -> Length: " << length << "\n";
        count++;
    }
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