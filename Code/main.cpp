#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>

#include "graphics.hpp"
#include "Mosaic.hpp"
#include "test.hpp"
#include "imageTest.hpp"

using namespace std;






int main() { 



    cout << "Hello From Image" << endl;

    bool verbose = false;
    image_test::Test my_test(verbose);
    my_test.runAllTests();




    // auto start = std::chrono::high_resolution_clock::now();


    // // cout << "Hello From Mosaic" << endl << endl;
    // cout << R"(
    // ███╗   ███╗ ██████╗ ███████╗ █████╗ ██╗ ██████╗     █████╗ ██████╗ ████████╗
    // ████╗ ████║██╔═══██╗██╔════╝██╔══██╗██║██╔════╝    ██╔══██╗██╔══██╗╚══██╔══╝
    // ██╔████╔██║██║   ██║███████╗███████║██║██║         ███████║██████╔╝   ██║   
    // ██║╚██╔╝██║██║   ██║╚════██║██╔══██║██║██║         ██╔══██║██╔══██╗   ██║   
    // ██║ ╚═╝ ██║╚██████╔╝███████║██║  ██║██║╚██████╗    ██║  ██║██║  ██║   ██║   
    // ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝ ╚═════╝    ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
    // )" << endl;
    // // credit :: https://patorjk.com/software/taag/#p=testall&f=Doom&t=Mosaic%20Art

    // // Load Object

    // mosaic_gen::Parameters params;
    // params.image_path = "../Images/flower.jpg";
    // params.results_dir = "../Results";
    // params.resize_factor = 1.5;
    // params.blur_kernel_size = 3;
    // params.blur_sigma = 1.4;
    // params.canny_threshold_1 = 50;
    // params.canny_threshold_2 = 100;
    // params.max_segment_angle_rad = 100 * M_PI / 180.0;
    // params.min_segment_length = 20;
    // params.segment_angle_window = 10;
    // params.tile_size = 10;
    // params.number_of_rings = 10;
    // params.step_size = 0.5 * params.tile_size;
    // params.min_intersection_distance = params.tile_size;
    // params.max_frontiers = 40;
    // params.flood_fill_neighbor_points = 16;
    // params.distance_from_center = params.tile_size * 1.5;
    // params.random_background_points = 50000;
    // params.tiles_per_frame = 20;
    // params.jitter_map.insert({4, 0});
    // params.jitter_map.insert({8, 20});
    // params.jitter_map.insert({12, 10});


    // mosaic_gen::Mosaic my_mosaic(params);
    



    // // RUN TESTS
    // MosaicTest::Test my_test(my_mosaic);
    // my_test.runAllTests();
    // my_test.runTimedProcess();


    // // RUN PREVIEW IN WINDOW
    // my_mosaic.resetData();
    // string window_name = "Mosaic Preview";
    // cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    // my_mosaic.setWindow(window_name);
    // my_mosaic.runAll();

    // auto end = std::chrono::high_resolution_clock::now();
    // chrono::duration<double> elapsed = end - start;
    // cout << "Main.cpp execution time: " << elapsed.count() << " s" << endl;

    // cv::imshow(window_name, my_mosaic.getCanvas());
    // cv::waitKey(0);





    



    return 0;
}




/*
TODO LIST

- replace drawSquare scalar function in all instances with vec3b ✅
- finish visualizations ✅
    - intersections ✅
    - fix vector field ✅
    - color frontiers ✅
    - show flood fill points along segment ✅
    - number segment placement ordering ✅
- utils/math/random file ✅
- color sampling options
- rename functions and variables
- standardize const and reference in function params
- use friend classes to move data back into mosiac private ✅
- vector of mats for saveGif/output

stretch goals
- more effieicnt getIntersections


*/