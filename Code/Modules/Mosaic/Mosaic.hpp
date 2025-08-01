#pragma once

#include "../Code/Utils/Image/Image.hpp"

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <opencv2/core.hpp>

using namespace std;

namespace mosaic_gen::test {
    class MosaicTest;
}

namespace mosaic_gen {


using image::Image;
using image::Size;
using image::Point;
using image::Color;
using image::Vec2d;

struct Parameters { 

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
    int tiles_per_frame;
    std::map<int, int> jitter_map;
    
    

};

struct TileInfo { 

    cv::Point center;
    double size;
    double theta_deg;
    int order;
    int frontier;

};




class Mosaic { 

    friend class test::MosaicTest;

    public: 

        Mosaic(const Parameters& p) : params(p) {};
        void setWindow(std::string name);
        void resetData();
        
        void runAll();
        cv::Mat getCanvas();


        void saveImage(const cv::Mat& image, const std::string& suffix);
        void saveGif(int tiles_per_frame, const std::string& suffix);
        void saveTileInfo(const std::string& suffix);
        

        string image_name;
        

    private: 

        int getJitter(int frontier);


        /*
        --------------------------------------
                        METHODS
        --------------------------------------
        */




        void contourPipeline();
        void selectStroke(int stroke_id);



















        // CONTOUR DETECTION PIPELINE
        void loadImage();
        void resizeOriginal();
        void grayImage();
        void blurImage();
        void cannyFilter();
        int detectContours();
        void rankContours();

        // PLACE TILES OVER CONTOURS
        void placeTileAllContours();
        void placeTileContour(int contour_id);
        void selectContour(int contour_id);
        cv::Point getRandomPointOnContour(int contour_id);
        double findBestTheta(cv::Point center, double size);
        bool isValidTile(cv::Point center, double size, double theta_deg);
        bool tileOverlapsMask(const cv::Point& center, double size, double theta_deg);
        bool tileInBounds(const cv::Point& center, double size);
        TileInfo placeTile(cv::Point center, double size, double theta_deg, int frontier=0, string text="");
        void renderTiles();
        std::vector<cv::Point> findRingIntersections(const cv::Mat& contour_image, const cv::Point2f& center, double size, double theta_deg);
        std::vector<cv::Point> filterUniqueIntersections(const std::vector<cv::Point>& intersection_points);
        
        // PLACE TILES FLOOD FILL
        void floodFill();
        std::vector<cv::Point> nextFrontierFromTile(cv::Point center, double theta_deg, double distance_from_center, int num_points);
        double findBestThetaTangentField(cv::Point center);
        std::tuple<cv::Vec2d, float> getTangentAtPoint(const cv::Point& point);
        void computeDistanceField();
        
        // PLACE TILES GAPS
        void gapFill();


        // RECREATE IMAGE FROM DISCRETE STATE
        void reconstructImage();
        cv::Vec3b sampleTileColor(const TileInfo& tile);


        /*
        --------------------------------------
                        DATA
        --------------------------------------
        */

        // settings for logic - should be user defined
        Parameters params;
        
        // store contours and their lengths
        std::vector<std::vector<Point>> segments;
        std::vector<double> segment_lengths;

        // discrete state representation for tiles
        std::vector<TileInfo> tiles_placed;
        std::vector<TileInfo> tiles_to_render; // to store tiles not yet rendered on canvas (will be cleared after render)

        // for the imshow operation
        std::string window_name;

        // image data various purposes
        Image original;
        Image resized;
        std::vector<std::vector<Point>> strokes;
        Image selected_stroke;





        // Image contours; // I question if we need this
        Image selected_contour;

        Image mask;
        Image canvas;

        // cv::Mat original;
        // cv::Mat resized;
        // cv::Mat grayscale;
        // cv::Mat blurred;
        // cv::Mat edges;
        // cv::Mat contours;
        // cv::Mat selected_contour;
        // cv::Mat distance_map;
        // cv::Mat gradX;
        // cv::Mat gradY;
        
        // cv::Mat mask;
        // cv::Mat canvas;


        // in case pca function fails
        const double ERROR_CODE_NO_VALID_THETA = -420.69;


};

}

