#ifndef MOSAIC_BUILDER_HPP
#define MOSAIC_BUILDER_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>

using namespace std;

namespace mosaic_gen {

class Mosaic { 

    public: 

        // param constructor
        Mosaic(const string& image_path);

        




        void resizeOriginal(double resize_factor);
        // void grayImage();
        // void blurImage(int kernel_size, double sigma);
        // void cannyFilter(int threshold_1, int threshold_2);
        // int detectContours(double max_segment_angle, int min_segment_length, int segment_angle_window);
        // void rankSegments();
        // void selectSegment(int k);
        // cv::Point getRandomPointOnSegment(int k);

        
        // void printColorToPixels();
        // void printColorLengths();
        // void printColorToPixelsK(int k);
        // void printColorLengthsK(int k);
        
        void saveImage(const cv::Mat& image, const std::string& output_dir, const std::string& suffix);


        cv::Mat original;
        cv::Mat resized;
        cv::Mat grayscale;
        cv::Mat blurred;
        cv::Mat edges;
        cv::Mat segmented;

        cv::Mat selected_segment;
        cv::Mat canvas;

        cv::Mat mask;
        std::string file_path;
        std::string image_name;


    private: 

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

        std::string vec3bToString(const cv::Vec3b& color);
        std::string pointToString(const cv::Point& pt);

        std::unordered_map<cv::Vec3b, std::vector<cv::Point>, Vec3bHash, Vec3bEqual> segment_pixels;
        std::vector<std::pair<cv::Vec3b, double>> segment_lengths;

};

}

#endif