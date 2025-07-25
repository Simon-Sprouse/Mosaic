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

        Mosaic(const Parameters& p);
        void setWindow(std::string name);
        void resetData();
        
        void runAll();
        cv::Mat getCanvas();

        void saveImage(const cv::Mat& image, const std::string& suffix);
        void saveGif(int tiles_per_frame, const std::string& suffix);
        void saveTileInfo(const std::string& suffix);
        

        string image_name;
        

    private: 


        /*
        --------------------------------------
                        METHODS
        --------------------------------------
        */

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
        std::vector<std::vector<cv::Point>> segment_points;
        std::vector<double> segment_lengths;

        // discrete state representation for tiles
        std::vector<TileInfo> tiles_placed;
        std::vector<TileInfo> tiles_to_render; // to store tiles not yet rendered on canvas (will be cleared after render)

        // for the imshow operation
        std::string window_name;

        // image data various purposes
        cv::Mat original;
        cv::Mat resized;
        cv::Mat grayscale;
        cv::Mat blurred;
        cv::Mat edges;
        cv::Mat contours;
        cv::Mat selected_contour;
        cv::Mat distance_map;
        cv::Mat gradX;
        cv::Mat gradY;
        cv::Mat canvas;
        cv::Mat mask;

        // in case pca function fails
        const double ERROR_CODE_NO_VALID_THETA = -420.69;


};

}

