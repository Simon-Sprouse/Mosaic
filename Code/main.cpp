#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>



#include "../Test/Utils/ImageProcess/ImageProcessTest.hpp"
#include "../Test/Utils/Geometry/geometryTest.hpp"
#include "../Test/Utils/Graphics/graphicsTest.hpp"
#include "../Test/Modules/Mosaic/MosaicTest.hpp"

#include "Utils/Io/io.hpp"
#include "Modules/Mosaic/Mosaic.hpp"



using namespace std;
using mosaic_gen::Mosaic;



std::vector<uint8_t> loadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
}





int main() { 

    using image::Image;


    cout << "Hello From Mosaic Test" << endl;
    string image_path = "../Images/flower.jpg";

    auto data = loadFile(image_path);
    Image img = image::fromEncodedBuffer(data.data(), data.size());
    io::saveImage(img, "../Results/data.jpg");

    


    // Mosaic Test
    mosaic_gen::Parameters params;
    params.image_path = "../Images/flower.jpg";
    params.results_dir = "../Results";
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






    Mosaic my_mosaic(params);
    my_mosaic.loadImageFromBuffer(data.data(), data.size());
    my_mosaic.runAll();
    io::saveImage(my_mosaic.getCanvas(), "../Results/output.png");
    cout << "my_mosiac.getCanvas(): " << my_mosaic.getCanvas() << endl;













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








    // // RUN TESTS
    // mosaic_gen::test::MosaicTest my_mosaic_test(my_mosaic);
    // my_mosaic_test.runAllTests();
    // my_mosaic_test.runTimedProcess();


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




    // // Image class tests
    // bool verbose = false;
    // image::test::ImageTest my_image_test(verbose);
    // my_image_test.runAllTests();



    // // Image process tests
    // image::process::test::ProcessTest my_process_test(image_path);
    // my_process_test.runAllTests();


    // // Geometry tests
    // Geometry::test::GeometryTest my_geometry_test("../Images/flower.jpg");
    // my_geometry_test.runAllTests();


    // // Graphics tests
    // Graphics::test::GraphicsTest my_graphics_test(image_path);
    // my_graphics_test.runAllTests();


    



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