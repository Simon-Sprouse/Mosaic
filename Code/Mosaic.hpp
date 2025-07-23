#pragma once


#include <string>
#include <vector>
#include <functional>
#include <opencv2/core.hpp>

using namespace std;

namespace MosaicTest {
    class Test;
}

namespace mosaic_gen {

struct HyperParameters { 

    string image_path;
    string results_dir;
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
    int min_intersection_distance;
    int max_frontiers;
    int flood_fill_neighbor_points;
    double distance_from_center;
    int random_background_points;

    std::function<int(int)> jitterFunc;
    int getJitter(int frontier) const {
        if (jitterFunc) {
            return jitterFunc(frontier);
        }
        return 0; // Default jitter
    }

};

struct TileInfo { 
    cv::Point center;
    double size;
    double theta_deg;
    int order;
    int frontier;
};




class Mosaic { 

    friend class MosaicTest::Test;

    public: 

        // param constructor
        Mosaic(const HyperParameters& hp);
        void resetData();

        void loadImage();
        void resizeOriginal();
        void grayImage();
        void blurImage();
        void cannyFilter();
        int detectContours();
        void rankSegments();
        void selectSegment(int k);
        cv::Point getRandomPointOnSegment(int k);
        bool tileInBounds(const cv::Point& center, double tileSize);
        bool tileOverlapsMask(const cv::Point& center, double tileSize, double rotationDegrees);
        bool isValidTile(cv::Point center, double tileSize, double theta_deg);
        double findBestTheta(cv::Point center, double size);
        TileInfo placeTile(cv::Point center, double size, double theta_deg, int frontier=0, string text="");
        void placeTileSegment(int k);
        void placeTileAllSegments();
        std::vector<cv::Point> findTileEdgeIntersections(const cv::Mat& segment_image, const cv::Point2f& center, double tileSize, double rotationDegrees);
        std::vector<cv::Point> filterUniqueIntersections(const std::vector<cv::Point>& inputPoints);
        cv::Vec3b sampleTileColor(const TileInfo& tile);
        void reconstructPlacedTiles();
      


        void computeDistanceField();
        std::tuple<cv::Vec2d, float> sampleTangentPoint(const cv::Point& pt);
        double findBestThetaTangentField(cv::Point center);
        
        std::vector<cv::Point> getFloodFillPoints2(cv::Point center, double theta_deg, double distance_from_center, int num_points);
        void showFloodFillPoints();
        void fillGapsRandom();
        
      
        void saveImage(const cv::Mat& image, const std::string& suffix);
        void saveGif(int tilesPerFrame, const std::string& suffix);
        void saveTileInfo(const std::string& suffix);

        string image_name;
        

        

    private: 

        HyperParameters params;
        
        std::vector<std::vector<cv::Point>> segments;
        std::vector<double> segment_lengths;

        std::vector<TileInfo> tiles_placed;

        cv::Mat original;
        cv::Mat resized;
        cv::Mat grayscale;
        cv::Mat blurred;
        cv::Mat edges;
        cv::Mat segmented;
        cv::Mat distance;
        cv::Mat gradX;
        cv::Mat gradY;
        cv::Mat selected_segment;
        cv::Mat canvas;
        cv::Mat mask;



        const double ERROR_CODE_NO_VALID_THETA = -420.69;



};

}

