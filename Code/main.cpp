#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "graphics.hpp"

#include "Mosaic.hpp"

using namespace std;
namespace fs = std::__fs::filesystem;


using mosaic_gen::Mosaic;




int main() { 

    auto start = chrono::high_resolution_clock::now();

    cout << "Hello From Mosaic" << endl;

    const string image_path = "../Images/flower.jpg";
    string results_dir = "../Results";


    mosaic_gen::HyperParameters params;
    params.resize_factor = 0.8;
    params.blur_kernel_size = 3;
    params.blur_sigma = 1.4;
    params.canny_threshold_1 = 50;
    params.canny_threshold_2 = 100;
    params.max_segment_angle_rad = 100 * M_PI / 180.0;
    params.min_segment_length = 20;
    params.segment_angle_window = 10;
    params.tile_size = 10;
    params.number_of_rings = 40;
    params.step_size = 0.5 * params.tile_size;

    

    // Load Image

    Mosaic my_mosaic(image_path);
    cout << "Loaded image: " << my_mosaic.image_name << endl;
    cout << "Original dimensions: " << my_mosaic.original.size() << endl;


    // Set hyperparameters
    my_mosaic.setParameters(params);


    // Resize Image

    my_mosaic.resizeOriginal();
    my_mosaic.saveImage(my_mosaic.resized, results_dir, "resized");
    cout << "Resized image to size: " << my_mosaic.resized.size() << endl;

    // Grayscale Image
    my_mosaic.grayImage();
    my_mosaic.saveImage(my_mosaic.grayscale, results_dir, "gray");

    // Blur Image
    my_mosaic.blurImage();
    my_mosaic.saveImage(my_mosaic.blurred, results_dir, "blurred");

    // Canny Filter
    my_mosaic.cannyFilter();
    my_mosaic.saveImage(my_mosaic.edges, results_dir, "canny_edges");

    // Detect Contours
    int contour_count = my_mosaic.detectContours();
    my_mosaic.saveImage(my_mosaic.segmented, results_dir, "segmented_edges");
    cout << "Detected: " << contour_count << " edges" << endl;


    // Rank Segments
    my_mosaic.rankSegments();

   
    // Place tiles on all segments
    my_mosaic.placeTileAllSegments();
    my_mosaic.saveImage(my_mosaic.mask, results_dir, "mask");



    // show flood fill points
    my_mosaic.showFloodFillPoints();
    my_mosaic.saveImage(my_mosaic.canvas, results_dir, "flood_fill_points");
   


    // sample points
    my_mosaic.placeTileAllBackground();
    my_mosaic.saveImage(my_mosaic.canvas, results_dir, "samples");

    // store placed tiles and reconstruct
    my_mosaic.reconstructPlacedTiles();
    my_mosaic.saveImage(my_mosaic.canvas, results_dir, "reconstruction");





    // distance field
    my_mosaic.computeDistanceField();
    cv::Mat distance_visual;
    my_mosaic.distance.convertTo(distance_visual, CV_8U, 255.0 / cv::norm(my_mosaic.distance, cv::NORM_INF));
    my_mosaic.saveImage(distance_visual, results_dir, "distance_field");

    // tangent field
    my_mosaic.sampleTangentField();
    my_mosaic.saveImage(my_mosaic.vector_field, results_dir, "vector_field");


    
  


    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end - start;
    cout << "Time to complete: " << elapsed_time.count() << " seconds" << endl;

    return 0;
}