#include <iostream>
#include <chrono>
#include <cmath>
#include <string>


#include "../Test/Utils/ImageProcess/ImageProcessTest.hpp"
#include "../Test/Utils/Geometry/geometryTest.hpp"
#include "../Test/Utils/Graphics/graphicsTest.hpp"
#include "../Test/Modules/Mosaic/MosaicTest.hpp"

#include "Utils/Image/io.hpp"
#include "Modules/Mosaic/Mosaic.hpp"



int main() { 

    using namespace std;
    using image::Image;
    using mosaic_gen::Mosaic;


    string file_system_image_path = "../Images/flower.jpg";
    string file_system_save_dir = "../Results/";

    mosaic_gen::Parameters params;
    params.resize_factor = 0.5;
    params.blur_kernel_size = 3;
    params.blur_sigma = 1.4;
    params.canny_threshold_1 = 50;
    params.canny_threshold_2 = 100;
    params.max_segment_angle_rad = 100 * M_PI / 180.0; // TODO why is this in rad
    params.min_segment_length = 20;
    params.segment_angle_window = 10;
    params.tile_size = 10;
    params.number_of_rings = 3;
    params.initial_step = 1.5 * params.tile_size;
    params.step_size = 0.25 * params.tile_size;
    params.min_intersection_distance = 0.25 * params.tile_size;
    params.max_frontiers = 7;
    params.flood_fill_neighbor_points = 4;
    params.distance_from_center = 1.5 * params.tile_size;
    params.random_background_points = 50000;
    params.canny_resize_factor = 0.4;
    // params.tiles_per_frame = 20;
    // params.jitter_map.insert({4, 0});
    // params.jitter_map.insert({8, 1});
    // params.jitter_map.insert({12, 10});

    int squeeb = 42;



    mosaic_gen::test::MosaicTest my_mosaic_test(params, file_system_image_path, file_system_save_dir);
    my_mosaic_test.runAllTests();



    return 0;




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




    Image img = image::io::loadImageFileSystem(file_system_image_path);
    cout << "Loaded image from: " << file_system_image_path << endl;
    cout << "Original Dimensions: " << img.size() << endl;

    Mosaic my_mosaic(params);
    my_mosaic.loadExistingImage(img);
    my_mosaic.runAll();

    cout << "Results Dimensions: " << my_mosaic.getCanvas().size() << endl;
    image::io::saveImageFileSystem(my_mosaic.getCanvas(), "../Results/result.jpg");




    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Main.cpp execution time: " << elapsed.count() << " s" << endl;






    return 0;
}


