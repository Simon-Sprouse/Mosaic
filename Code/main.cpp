#include <iostream>
#include <chrono>
#include <cmath>
#include <string>


#include "../Test/Utils/ImageProcess/ImageProcessTest.hpp"
#include "../Test/Utils/Geometry/geometryTest.hpp"
#include "../Test/Utils/Graphics/graphicsTest.hpp"
#include "../Test/Modules/Mosaic/MosaicTest.hpp"

#include "Utils/Io/io.hpp"
#include "Modules/Mosaic/Mosaic.hpp"



int main() { 

    using namespace std;
    using image::Image;
    using mosaic_gen::Mosaic;


    string file_system_image_path = "../Images/flower.jpg";
    string file_system_results_dir = "../Results";

    mosaic_gen::Parameters params;
    params.resize_factor = 0.8;
    params.blur_kernel_size = 3;
    params.blur_sigma = 1.4;
    params.canny_threshold_1 = 50;
    params.canny_threshold_2 = 100;
    params.max_segment_angle_rad = 100 * M_PI / 180.0; // TODO why is this in rad
    params.min_segment_length = 20;
    params.segment_angle_window = 10;
    params.tile_size = 10;
    params.number_of_rings = 5;
    params.step_size = 1 * params.tile_size;
    params.min_intersection_distance = 1.5 * params.tile_size;
    params.max_frontiers = 20;
    params.flood_fill_neighbor_points = 16;
    params.distance_from_center = 1.5 * params.tile_size;
    params.random_background_points = 50000;
    params.tiles_per_frame = 20;
    params.jitter_map.insert({4, 0});
    params.jitter_map.insert({8, 1});
    params.jitter_map.insert({12, 10});


    // // cout << "Hello From Mosaic" << endl << endl;
    cout << R"(
    ███╗   ███╗ ██████╗ ███████╗ █████╗ ██╗ ██████╗     █████╗ ██████╗ ████████╗
    ████╗ ████║██╔═══██╗██╔════╝██╔══██╗██║██╔════╝    ██╔══██╗██╔══██╗╚══██╔══╝
    ██╔████╔██║██║   ██║███████╗███████║██║██║         ███████║██████╔╝   ██║   
    ██║╚██╔╝██║██║   ██║╚════██║██╔══██║██║██║         ██╔══██║██╔══██╗   ██║   
    ██║ ╚═╝ ██║╚██████╔╝███████║██║  ██║██║╚██████╗    ██║  ██║██║  ██║   ██║   
    ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝ ╚═════╝    ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
    )" << endl;
    // credit :: https://patorjk.com/software/taag/#p=testall&f=Doom&t=Mosaic%20Art
    auto start = std::chrono::high_resolution_clock::now();





    Mosaic my_mosaic(params);
    Image img = io::loadImageFileSystem(file_system_image_path);
    cout << "img.size(): " << img.size() << endl;
    my_mosaic.loadExistingImage(img);
    my_mosaic.runAll();
    cout << "my_mosaic.getCanvas().size(): " << my_mosaic.getCanvas().size() << endl;
    io::saveImage(my_mosaic.getCanvas(), "../Results/output.png");




    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Main.cpp execution time: " << elapsed.count() << " s" << endl;






    return 0;
}


