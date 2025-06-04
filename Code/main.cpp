#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "image_process.hpp"

using namespace std;
namespace fs = std::__fs::filesystem;
using ImageProcess::ImageState;





int main() { 

    auto start = chrono::high_resolution_clock::now();

    cout << "Hello From Mosaic" << endl;

    string image_path = "../Images/flower.jpg";
    string results_dir = "../Results";

    // parameters
    double RESIZE_FACTOR = 0.8;
    int BLUR_KERNEL_SIZE = 3;
    double BLUR_SIGMA = 1.4;
    int CANNY_THRESHOLD_1 = 50;
    int CANNY_THRESHOLD_2 = 100;
    double MAX_SEGMENT_ANGLE = 40 * M_PI / 180.0;
    int MIN_SEGMENT_LENGTH = 20;
    int SEGMENT_ANGLE_WINDOW = 10;

    // Load Image
    ImageState img_state(image_path);
    ImageProcess::saveImage(img_state.original, results_dir, img_state.file_name, "original");
    cout << "Loaded image: " << img_state.file_name << endl;
    cout << "Original dimensions: " << img_state.original.size() << endl;


    // Resize Image
    ImageProcess::resizeImage(img_state, RESIZE_FACTOR);
    ImageProcess::saveImage(img_state.resized, results_dir, img_state.file_name, "rescaled");
    cout << "Resized image to size: " << img_state.resized.size() << endl;

    // Grayscale Image
    ImageProcess::grayImage(img_state);
    ImageProcess::saveImage(img_state.grayscale, results_dir, img_state.file_name, "grayscale");

    // Blur Image
    ImageProcess::blurImage(img_state, BLUR_KERNEL_SIZE, BLUR_SIGMA);
    ImageProcess::saveImage(img_state.blurred, results_dir, img_state.file_name, "blurred");

    // Canny Filter
    ImageProcess::cannyFilter(img_state, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2);
    ImageProcess::saveImage(img_state.edges, results_dir, img_state.file_name, "canny_edges");

    // Detect Contours
    int contour_count = ImageProcess::detectContours(img_state, MAX_SEGMENT_ANGLE, MIN_SEGMENT_LENGTH, SEGMENT_ANGLE_WINDOW);
    ImageProcess::saveImage(img_state.segmented, results_dir, img_state.file_name, "segmented");
    cout << "Found " << contour_count << " contour segments" << endl;

    // Rank Segments
    ImageProcess::rankSegments(img_state);
    ImageProcess::printColorToPixelsK(img_state.segment_pixels);
    ImageProcess::printColorLengthsK(img_state.segment_lengths);

    // Select Segment
    ImageProcess::selectSegment(img_state, 1);
    ImageProcess::saveImage(img_state.selected_segment, results_dir, img_state.file_name, "selected_segment");

    // Draw Square



    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end - start;
    cout << "Time to complete: " << elapsed_time.count() << " seconds" << endl;

    return 0;
}