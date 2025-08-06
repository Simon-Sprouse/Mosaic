#include "ImageProcessTest.hpp"

#include "../../../Code/Utils/Image/io.hpp"
#include "../../../Code/Utils/ImageProcess/imageProcess.hpp"

namespace image::process::test { 






void ProcessTest::testLoadImage() { 
    Image img = io::loadImageFileSystem(image_path_);
    io::saveImageFileSystem(img, "../Results/original.jpg");


    
}






void ProcessTest::testComputeDistanceField() { 


    // necessary progress
    Image src_img = io::loadImageFileSystem(image_path_);

    Image gray_img;
    grayscale(src_img, gray_img);

    Image blur_img;
    Size kernel_size = Size(3, 3);
    double blur_sigma = 1.4;
    gaussianBlur(gray_img, blur_img, kernel_size, blur_sigma);

    Image canny_img;
    int canny_threshold_1 = 50;
    int canny_threshold_2 = 100;
    cannyFilter(blur_img, canny_img, canny_threshold_1, canny_threshold_2);




    // Image distance_map;
    std::vector<float> distance_field = computeDistanceField(canny_img);
    Image distance_field_img = floatMapToGrayscaleImage(distance_field, canny_img.size());

    io::saveImageFileSystem(distance_field_img, "../Results/distance_field.jpg");


}




void ProcessTest::testDistanceGradients() { 

        // necessary progress
        Image src_img = io::loadImageFileSystem(image_path_);

        Image gray_img;
        grayscale(src_img, gray_img);
    
        Image blur_img;
        Size kernel_size = Size(3, 3);
        double blur_sigma = 1.4;
        gaussianBlur(gray_img, blur_img, kernel_size, blur_sigma);
    
        Image canny_img;
        int canny_threshold_1 = 50;
        int canny_threshold_2 = 100;
        cannyFilter(blur_img, canny_img, canny_threshold_1, canny_threshold_2);
    
    
    
    
        // Image distance_map;
        std::vector<float> distance_field = computeDistanceField(canny_img);
        Image distance_field_img = floatMapToGrayscaleImage(distance_field, canny_img.size());

        std::vector<float> grad_x;
        std::vector<float> grad_y;
        computeSobelGradients(distance_field, canny_img.size(), grad_x, grad_y);

        Image grad_x_img = floatMapToGrayscaleImage(grad_x, canny_img.size());
        Image grad_y_img = floatMapToGrayscaleImage(grad_y, canny_img.size());
    
        io::saveImageFileSystem(grad_x_img, "../Results/grad_x.jpg");
        io::saveImageFileSystem(grad_y_img, "../Results/grad_y.jpg");
        
    
}












void ProcessTest::runAllTests() { 
    cout << "RUNNING TESTS... " << endl;
    timeFunctionBar();

    chrono::duration<double> total_time(0.0);

    // call test functions
    total_time += timeFunction("Load Image", [&]() {(testLoadImage());});
    total_time += timeFunction("Compute Distance Map", [&]() {(testComputeDistanceField());});
    total_time += timeFunction("Compute Distance Gradients", [&]() {(testDistanceGradients());});




    timeFunctionBar();
    printTotalTime(total_time);

    cout << endl;
}







}

