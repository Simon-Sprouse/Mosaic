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

        // void resizeOriginal(double resize_factor);



    private: 

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
        std::string file_path;
        std::string image_name;

};

}

#endif