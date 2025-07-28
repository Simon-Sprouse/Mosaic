#include "imageProcessTest.hpp"
#include "../Io/io.hpp"

namespace image::process::test { 


void ProcessTest::loadImage() { 
    Image img = io::loadImage(image_path_);
    io::saveImage(img, "../Results/original.jpg");
}

void ProcessTest::squeebTest() { 
    int squeeb = 2;
}



void ProcessTest::runAllTests() { 

    cout << "RUNNING TESTS... " << endl;
    printHorizontalBar();

    chrono::duration<double> total_time(0.0);

    // call test functions
    total_time += timeFunction("Load Image", [&]() {loadImage();});
    total_time += timeFunction("Squeeb Test", [&]() {squeebTest();});




    printHorizontalBar();
    printTotalTime(total_time);

    cout << endl;




}


}