#include <iostream>
#include <chrono>
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

    // Load Image
    ImageState img_state(image_path);
    cout << "Loaded image: " << img_state.file_name << endl;
    cout << "Original dimensions: " << img_state.original.size() << endl;
    ImageProcess::saveImage(img_state.original, results_dir, img_state.file_name, "original");

    // Resize Image
    ImageProcess::resizeImage(img_state, RESIZE_FACTOR);
    cout << "Resized image to size: " << img_state.resized.size() << endl;
    ImageProcess::saveImage(img_state.resized, results_dir, img_state.file_name, "rescaled");


    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end - start;
    cout << "Time to complete: " << elapsed_time.count() << " seconds" << endl;

    return 0;
}