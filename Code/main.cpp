#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>

#include "graphics.hpp"
#include "Mosaic.hpp"
#include "test.hpp"

using namespace std;
namespace fs = std::__fs::filesystem;


using mosaic_gen::Mosaic;




int main() { 


    // cout << "Hello From Mosaic" << endl << endl;
    cout << R"(
    ███╗   ███╗ ██████╗ ███████╗ █████╗ ██╗ ██████╗     █████╗ ██████╗ ████████╗
    ████╗ ████║██╔═══██╗██╔════╝██╔══██╗██║██╔════╝    ██╔══██╗██╔══██╗╚══██╔══╝
    ██╔████╔██║██║   ██║███████╗███████║██║██║         ███████║██████╔╝   ██║   
    ██║╚██╔╝██║██║   ██║╚════██║██╔══██║██║██║         ██╔══██║██╔══██╗   ██║   
    ██║ ╚═╝ ██║╚██████╔╝███████║██║  ██║██║╚██████╗    ██║  ██║██║  ██║   ██║   
    ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝ ╚═════╝    ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
    )" << endl;
    // credit :: https://patorjk.com/software/taag/#p=testall&f=Doom&t=Mosaic%20Art

    // Load Object

    mosaic_gen::HyperParameters params;
    params.image_path = "../Images/flower.jpg";
    // params.image_path = "/Users/simonsprouse/Desktop/prayer.png";
    // params.image_path = "/Users/simonsprouse/Desktop/CSCE_448/final/dunes.jpg";
    params.results_dir = "../Results";
    params.resize_factor = 1.5;
    params.blur_kernel_size = 3;
    params.blur_sigma = 1.4;
    params.canny_threshold_1 = 50;
    params.canny_threshold_2 = 100;
    params.max_segment_angle_rad = 100 * M_PI / 180.0;
    params.min_segment_length = 20;
    params.segment_angle_window = 10;
    params.tile_size = 10;
    params.number_of_rings = 10;
    params.step_size = 0.5 * params.tile_size;
    params.min_intersection_distance = params.tile_size;
    params.max_frontiers = 40;
    params.flood_fill_neighbor_points = 16;
    params.distance_from_center = params.tile_size * 1.5;

    params.random_background_points = 50000;
    params.jitterFunc = [](int frontier) -> int {
        if (frontier < 4) return 0;
        if (frontier < 8) return 2;
        return 10;
    };

    Mosaic my_mosaic(params);
    my_mosaic.loadImage();
    my_mosaic.resizeOriginal();
    cout << "Loaded image: " << my_mosaic.image_name << endl;
    cout << "Original dimensions: " << my_mosaic.original.size() << endl;
    cout << "Resized image to size: " << my_mosaic.resized.size() << endl;
    cout << endl;



    // RUN TESTS
    Test::runAllTests(my_mosaic);




    // RUN PROCESS
    Test::runTimedProcess(my_mosaic);
   

    return 0;
}




/*
TODO LIST

- replace drawSquare scalar function in all instances with vec3b ✅
- finish visualizations
    - intersections ✅
    - fix vector field ✅
    - color frontiers ✅
    - show flood fill points along segment ✅
    - number segment placement ordering ✅
- utils/math/random file
- color sampling options
- rename functions and variables
- standardize const and reference in function params
- use friend classes to move data back into mosiac private
- vector of mats for saveGif/output

stretch goals
- more effieicnt getIntersections


*/