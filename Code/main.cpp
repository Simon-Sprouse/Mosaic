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

    // parameters
    double RESIZE_FACTOR = 0.8;
    int BLUR_KERNEL_SIZE = 3;
    double BLUR_SIGMA = 1.4;
    int CANNY_THRESHOLD_1 = 50;
    int CANNY_THRESHOLD_2 = 100;
    double MAX_SEGMENT_ANGLE_RAD = 100 * M_PI / 180.0;
    int MIN_SEGMENT_LENGTH = 20;
    int SEGMENT_ANGLE_WINDOW = 10;


    // Load Image

    Mosaic my_mosaic(image_path);
    cout << "Loaded image: " << my_mosaic.image_name << endl;
    cout << "Original dimensions: " << my_mosaic.original.size() << endl;

    // Resize Image

    my_mosaic.resizeOriginal(RESIZE_FACTOR);
    my_mosaic.saveImage(my_mosaic.resized, results_dir, "resized");
    cout << "Resized image to size: " << my_mosaic.resized.size() << endl;

    // Grayscale Image
    my_mosaic.grayImage();
    my_mosaic.saveImage(my_mosaic.grayscale, results_dir, "gray");

    // Blur Image
    my_mosaic.blurImage(BLUR_KERNEL_SIZE, BLUR_SIGMA);
    my_mosaic.saveImage(my_mosaic.blurred, results_dir, "blurred");

    // Canny Filter
    my_mosaic.cannyFilter(CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);
    my_mosaic.saveImage(my_mosaic.edges, results_dir, "canny_edges");

    // Detect Contours
    int contour_count = my_mosaic.detectContours(MAX_SEGMENT_ANGLE_RAD, MIN_SEGMENT_LENGTH, SEGMENT_ANGLE_WINDOW);
    my_mosaic.saveImage(my_mosaic.segmented, results_dir, "segmented_edges");
    cout << "Detedted: " << contour_count << " edges" << endl;



    int k = 1;

    // Rank Segments
    my_mosaic.rankSegments();
    my_mosaic.printColorToPixelsK(k);
    my_mosaic.printColorLengthsK(k);

    // Select Segment
    my_mosaic.selectSegment(k);
    my_mosaic.saveImage(my_mosaic.selected_segment, results_dir, "selected_segment");


    // Get random point on segment
    cv::Point my_point = my_mosaic.getRandomPointOnSegment(k);
    cout << "Random Point: " << my_point << endl;

    // // Draw Square
    // my_mosaic.drawSquareRandomPoint(k);
    // my_mosaic.saveImage(my_mosaic.canvas, results_dir, "draw_test");

    // // Test Reward Function
    // my_mosaic.placeTile(k);
    // my_mosaic.saveImage(my_mosaic.canvas, results_dir, "reward_test");


    // Complete One BFS Frontier
    my_mosaic.placeTileSegment(k);
    my_mosaic.saveImage(my_mosaic.canvas, results_dir, "placeTileSegment_test");
    my_mosaic.saveImage(my_mosaic.mask, results_dir, "mask");
  


    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end - start;
    cout << "Time to complete: " << elapsed_time.count() << " seconds" << endl;

    return 0;
}