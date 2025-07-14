#ifndef MOSAIC_BUILDER_HPP
#define MOSAIC_BUILDER_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>

using namespace std;

namespace mosaic_gen {

struct HyperParameters { 

    double resize_factor;
    int blur_kernel_size;
    double blur_sigma;
    int canny_threshold_1;
    int canny_threshold_2;
    double max_segment_angle_rad;
    int min_segment_length;
    int segment_angle_window;
    int tile_size;
    int number_of_rings;
    int step_size;

};

struct TileInfo { 
    cv::Point center;
    double size;
    double theta_deg;
    int order;
};

class Mosaic { 

    public: 

        // param constructor
        Mosaic(const string& image_path);

        void setParameters(const HyperParameters& hp);

        void resizeOriginal();
        void grayImage();
        void blurImage();
        void cannyFilter();
        int detectContours();
        void rankSegments();
        void selectSegment(int k);
        cv::Point getRandomPointOnSegment(int k);
        void drawSquareRandomPoint(int k); // test
        bool tileOverlapsMask(const cv::Point& center, double tileSize, double rotationDegrees);
        double placeTile(cv::Point center, double size, string text="");
        void placeTileSegment(int k);
        void placeTileAllSegments();
        std::vector<cv::Point> findTileEdgeIntersections(const cv::Mat& segment_image, const cv::Point2f& center, double tileSize, double rotationDegrees);
        std::vector<cv::Point> filterUniqueIntersections(const std::vector<cv::Point>& inputPoints);
        void reconstructPlacedTiles();
        
        void printColorToPixels();
        void printColorLengths();
        void printColorToPixelsK(int k);
        void printColorLengthsK(int k);
        
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

        HyperParameters params;
        std::vector<TileInfo> tiles_placed;

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