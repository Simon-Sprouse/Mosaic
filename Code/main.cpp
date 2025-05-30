#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_process.hpp"

using namespace std;
namespace fs = std::__fs::filesystem;
using ImageProcess::ImageState;



int main() { 
    cout << "Hello From Mosaic" << endl;

    string imagePath = "../Images/flower.jpg";

    ImageState img_state = ImageProcess::loadImage(imagePath);

    cout << img_state.name << endl;

    string output_dir = "../Results";


    ImageProcess::saveImage(img_state.original, output_dir, img_state.name);


    return 0;
}