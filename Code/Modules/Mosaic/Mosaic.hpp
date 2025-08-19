#pragma once

#include "../../Utils/Image/Image.hpp"

#include <queue>
#include <stack>
#include <string>
#include <vector>




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



    // string image_path;
    // string results_dir;
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
    double initial_step;
    int step_size;
    int min_intersection_distance;
    int max_frontiers;
    int flood_fill_neighbor_points;
    double distance_from_center;
    int random_background_points;
    double canny_resize_factor;
    // int tiles_per_frame;
    // std::map<int, int> jitter_map;
    
    

};

struct TileInfo { 

    Point center;
    double size;
    double theta_deg;
    Color color;
    int order;
    int frontier;

};




class Mosaic { 

    friend class test::MosaicTest;

    public: 

        Mosaic(const Parameters& p) : params(p) {};
        // void setWindow(std::string name);
        // void resetData();
        
        void loadImageFromBuffer(const uint8_t* data, size_t size);
        void loadImageFromVector(const std::vector<uint8_t>& buffer);

        void loadExistingImage(const Image& img);

        void contourPipeline();
        void runAll();
        Image getCanvas();
        Image* getCanvasPtr();
        Image* getDebugCanvasPtr();
        Image* getStrokesImagePtr();
        Image* getOriginalImagePtr();
        Image getContourImage();
        uint8_t* getRawData();
        bool empty();
        Size size();

        bool stepK(int k);
        void renderImageRange(int start, int num_tiles);
        int getRenderPointer();
        void setRenderPointer(int start);
        void resetCanvas();
        void renderDebugImageRange(int start, int num_tiles);

        void setParameters(Parameters p);
        void clearData();
        void resizeOriginal();


        void setDebugMode(bool state);

        string image_name;
        

    private: 

        


        /*
        --------------------------------------
                        METHODS
        --------------------------------------
        */




        
        void selectStroke(int stroke_id);

        bool isValidTile(Point center, double size, double theta_deg);
        bool tileInBounds(const Point& center);
        bool tileOverlapsMask(const Point& center, double tile_size, double theta_deg);
        TileInfo placeTile(Point center, double size, double theta_deg, int frontier=0, string text="");

        Point getRandomPointOnStroke(int stroke_id);

        double findBestTheta(Point center, double size);
        std::vector<Point> findNonZeroInRadius(const Image& src, const Point& center, int radius);

        std::vector<Point> findPointsMultipleRings(const Point& center, double theta_deg);
        std::vector<Point> findRingIntersections(const Point& center, double ring_size, double theta_deg, int thickness);


        void placeTilesAlongStroke(int stroke_id);
        void placeTilesAllStrokes();


        void floodFill();
        int getJitter(int frontier);
        void computeDistanceField();
        double findThetaTangent(Point center);

        void gapFill();

        void reconstructImage();
        

        void reconstructShowFrontiers();
        Color sampleTileColor(Point center, double size, double theta_deg);

        bool stepOnce();
        
        

     


        /*
        --------------------------------------
                        DATA
        --------------------------------------
        */

        // settings for logic - should be user defined
        Parameters params;
       
        std::vector<TileInfo> tiles_placed;
        
        // image data various purposes
        Image original;
        Image resized;
        Image canny;
        Image strokes_image;
        std::vector<std::vector<Point>> strokes;
        Image selected_stroke;

        std::vector<float> grad_x;
        std::vector<float> grad_y;

        Image mask;
        Image canvas;
        Image debugCanvas; // TODO unify this logic with canvas somehow



        std::stack<Point> strokePointsStack;
        std::queue<Point> floodPointsQueue;
        std::vector<Point> gapPointsVector;

        int strokes_completed = 0;
        int frontiers_completed = 0;
        bool gaps_calculated = false;

        // int num_tiles_rendered = 0; // used to avoid re-rendering entire tiles_placed vector every step;
        int render_pointer = 0;

        bool debug_mode = false;


        // in case pca function fails
        const double ERROR_CODE_NO_VALID_THETA = -420.69;


};

}

