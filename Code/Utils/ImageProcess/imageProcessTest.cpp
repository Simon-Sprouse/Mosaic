#include "imageProcessTest.hpp"
#include "../Io/io.hpp"
#include "imageProcess.hpp"

namespace image::process::test { 


void ProcessTest::loadImage() { 
    Image img = io::loadImage(image_path_);
    io::saveImage(img, "../Results/original.jpg");
}

void ProcessTest::squeebTest() { 
    int squeeb = 2;
}


void ProcessTest::testResize() { 

    // necessary progress
    Image src_img = io::loadImage(image_path_);
    Image dest_img;

    // resize(w, h)
    int w = 400;
    int h = 400;
    image::process::resize(src_img, dest_img, w, h);
    io::saveImage(dest_img, "../Results/resize_int.jpg");

    // resize(size)
    Size new_size(600, 1200);
    image::process::resize(src_img, dest_img, new_size);
    io::saveImage(dest_img, "../Results/resize_size.jpg");

    // resize(ratio)
    double ratio = 2.7;
    image::process::resize(src_img, dest_img, ratio);
    io::saveImage(dest_img, "../Results/resize_high_ratio.jpg");

    ratio = 0.2;
    image::process::resize(src_img, dest_img, ratio);
    io::saveImage(dest_img, "../Results/resize_low_ratio.jpg");


}

















void ProcessTest::runAllTests() { 

    cout << "RUNNING TESTS... " << endl;
    printHorizontalBar();

    chrono::duration<double> total_time(0.0);

    // call test functions
    total_time += timeFunction("Load Image", [&]() {loadImage();});
    total_time += timeFunction("Squeeb Test", [&]() {squeebTest();});
    total_time += timeFunction("Resize Image", [&]() {testResize();});




    printHorizontalBar();
    printTotalTime(total_time);

    cout << endl;




}


}