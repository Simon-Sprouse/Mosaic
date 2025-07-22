#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

using namespace std;

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
    int max_frontiers;
    int flood_fill_neighbor_points;
    int flood_fill_point_jitter;
    int random_background_points;

};

struct TileInfo { 
    cv::Point center;
    double size;
    double theta_deg;
    int order;
    int frontier;
};

class Mosaic { 

    public: 

        // param constructor
        Mosaic(const HyperParameters& hp);


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
        cv::Scalar sampleTileColor(const TileInfo& tile);
        void reconstructPlacedTiles();
        double randomDouble(double min_val, double max_val);
        std::vector<cv::Point> samplePointsGrid(const cv::Mat& image, int grid_size);
        std::vector<cv::Point> samplePointsRandom(const cv::Mat& image, int num_points);
        std::vector<cv::Point> jitterPoints(const std::vector<cv::Point>& input_points, int max_step, const cv::Size& image_size);


       
        void placeTileAllBackground();

        void computeDistanceField();
        std::tuple<cv::Vec2f, float> sampleTangentPoint(const cv::Point& pt);
        double findBestThetaTangentField(cv::Point center);
        
        std::vector<cv::Point> getFloodFillPoints2(cv::Point center, double theta_deg, double distance_from_center, int num_points);
        void showFloodFillPoints();
        
      
        void saveImage(const cv::Mat& image, const std::string& suffix);
        void saveGif(int tilesPerFrame, const std::string& suffix);
        void saveTileInfo(const std::string& suffix);


        string image_name;

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

    
        std::vector<std::vector<cv::Point>> segments;
        std::vector<double> segment_lengths;

        


    private: 



        HyperParameters params;
        std::vector<TileInfo> tiles_placed;

        const double ERROR_CODE_NO_VALID_THETA = -420.69;



};

}

