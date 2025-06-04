#ifndef IMAGE_PROCESS_HPP
#define IMAGE_PROCESS_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/imgcodecs.hpp>

namespace ImageProcess { 

    struct Vec3bHash {
        std::size_t operator()(const cv::Vec3b& color) const noexcept {
            return std::hash<int>()(
                (static_cast<int>(color[0]) << 16) |
                (static_cast<int>(color[1]) << 8) |
                (static_cast<int>(color[2]))
            );
        }
    };
    
    struct Vec3bEqual {
        bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const noexcept {
            return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
        }
    };
    
    
    struct ImageState { 
        cv::Mat original;
        cv::Mat resized;
        cv::Mat grayscale;
        cv::Mat blurred;
        cv::Mat edges;
        cv::Mat segmented;

        std::unordered_map<cv::Vec3b, std::vector<cv::Point>, Vec3bHash, Vec3bEqual> segment_pixels;
        std::vector<std::pair<cv::Vec3b, double>> segment_lengths;



        cv::Mat selected_segment;
        cv::Mat canvas;

        cv::Mat mask;
        std::string file_name;
        std::string file_path;

        // param constructor
        ImageState(const std::string& image_path);
    };

    
    
    void resizeImage(ImageState& state, double resize_factor);
    void grayImage(ImageState& state);
    void blurImage(ImageState& state, int kernel_size, double sigma);
    void cannyFilter(ImageState& state, int threshold_1, int threshold_2);
    int detectContours(ImageState& state, double max_segment_angle, int min_segment_length, int segment_angle_window);
    void rankSegments(ImageState& state);
    void selectSegment(ImageState& state, int k);
    cv::Point getRandomPointOnSegment(ImageState& state, int k);

    std::string vec3bToString(const cv::Vec3b& color);
    std::string pointToString(const cv::Point& pt);
    void printColorToPixels(const std::unordered_map<cv::Vec3b, std::vector<cv::Point>, ImageProcess::Vec3bHash, ImageProcess::Vec3bEqual>& color_to_pixels);
    void printColorLengths(const std::vector<std::pair<cv::Vec3b, double>>& color_lengths);
    void printColorToPixelsK(const std::unordered_map<cv::Vec3b, std::vector<cv::Point>, ImageProcess::Vec3bHash, ImageProcess::Vec3bEqual>& color_to_pixels);
    void printColorLengthsK(const std::vector<std::pair<cv::Vec3b, double>>& color_lengths);
    
    void saveImage(const cv::Mat& image, const std::string& output_dir, const std::string& output_name, const std::string& suffix);

    

}

#endif