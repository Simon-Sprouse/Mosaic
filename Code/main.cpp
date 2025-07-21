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
    // params.image_path = "/Users/simonsprouse/Desktop/Art/Screenshot 2024-10-05 at 9.58.54 PM.png";
    params.results_dir = "../Results";
    params.resize_factor = 1.2;
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
    params.max_frontiers = 40;
    params.flood_fill_neighbor_points = 16;
    params.flood_fill_point_jitter = 1;
    params.random_background_points = 50000;

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