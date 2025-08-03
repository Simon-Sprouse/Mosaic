// #include "imageProcessTest.hpp"
// #include "../Io/io.hpp"
// #include "imageProcess.hpp"

// #include <random>

// namespace image::process::test { 


// void ProcessTest::loadImage() { 
//     Image img = io::loadImage(image_path_);
//     io::saveImage(img, "../Results/original.jpg");
// }

// void ProcessTest::squeebTest() { 
//     int squeeb = 2;
// }


// void ProcessTest::testResize() { 

//     // necessary progress
//     Image src_img = io::loadImage(image_path_);
//     Image dest_img;

//     // resize(w, h)
//     int w = 400;
//     int h = 400;
//     image::process::resize(src_img, dest_img, w, h);
//     io::saveImage(dest_img, "../Results/resize_int.jpg");

//     // resize(size)
//     Size new_size(600, 1200);
//     image::process::resize(src_img, dest_img, new_size);
//     io::saveImage(dest_img, "../Results/resize_size.jpg");

//     // resize(ratio)
//     double ratio = 2.7;
//     image::process::resize(src_img, dest_img, ratio);
//     io::saveImage(dest_img, "../Results/resize_high_ratio.jpg");

//     ratio = 0.2;
//     image::process::resize(src_img, dest_img, ratio);
//     io::saveImage(dest_img, "../Results/resize_low_ratio.jpg");


// }



// void ProcessTest::testGrayscale() { 

//     // necessary progress
//     Image src_img = io::loadImage(image_path_);
//     Image dest_img;

//     image::process::grayscale(src_img, dest_img);
//     io::saveImage(dest_img, "../Results/grayscale.jpg");
// }



// void ProcessTest::testBlur() { 

//     // necessary progress
//     Image src_img = io::loadImage(image_path_);
//     Image dest_img;

//     Size kernel_size;
//     double blur_sigma;

//     // high sigma blur
//     kernel_size = Size(5, 5);
//     blur_sigma = 5;
//     image::process::gaussianBlur(src_img, dest_img, kernel_size, blur_sigma);
//     io::saveImage(dest_img, "../Results/high_sigma_blur.jpg");

//     // low sigma blur
//     kernel_size = Size(5, 5);
//     blur_sigma = 0.5;
//     image::process::gaussianBlur(src_img, dest_img, kernel_size, blur_sigma);
//     io::saveImage(dest_img, "../Results/low_sigma_blur.jpg");

//     // high sigma blur
//     kernel_size = Size(21, 21);
//     blur_sigma = 2;
//     image::process::gaussianBlur(src_img, dest_img, kernel_size, blur_sigma);
//     io::saveImage(dest_img, "../Results/large_kernel_blur.jpg");

//     // high sigma blur
//     kernel_size = Size(3, 3);
//     blur_sigma = 2;
//     image::process::gaussianBlur(src_img, dest_img, kernel_size, blur_sigma);
//     io::saveImage(dest_img, "../Results/small_kernel_blur.jpg");

//     // extreme blur
//     kernel_size = Size(31, 31);
//     blur_sigma = 20;
//     image::process::gaussianBlur(src_img, dest_img, kernel_size, blur_sigma);
//     io::saveImage(dest_img, "../Results/extreme_blur.jpg");



// }




// void ProcessTest::testSobel() { 

//     // necessary progress
//     Image src_img = io::loadImage(image_path_);
//     Image gray_img;
//     image::process::grayscale(src_img, gray_img);

//     Image dest_img_x;
//     Image dest_img_y;

//     image::process::sobelFilter(gray_img, dest_img_x, dest_img_y);
//     io::saveImage(dest_img_x, "../Results/sobel_grad_x.jpg");
//     io::saveImage(dest_img_y, "../Results/sobel_grad_y.jpg");


//     Image dest_sobel_visual;
//     image::process::visualizeSobel(dest_img_x, dest_img_y, dest_sobel_visual);
//     io::saveImage(dest_sobel_visual, "../Results/sobel_visual.jpg");


// }



// void ProcessTest::testCanny() { 

//     // necessary progress
//     Image src_img = io::loadImage(image_path_);
//     Image gray_img;
//     image::process::grayscale(src_img, gray_img);
//     Image blur_img;
//     Size kernel_size = Size(3, 3);
//     double blur_sigma = 1.4;
//     image::process::gaussianBlur(gray_img, blur_img, kernel_size, blur_sigma);

//     Image canny_img;
//     int canny_threshold_1 = 50;
//     int canny_threshold_2 = 100;
//     image::process::cannyFilter(blur_img, canny_img, canny_threshold_1, canny_threshold_2);
//     io::saveImage(canny_img, "../Results/canny_filter.jpg");





// }



// void ProcessTest::testContours() { 

//     // necessary progress
//     Image src_img = io::loadImage(image_path_);
//     Image gray_img;
//     image::process::grayscale(src_img, gray_img);
//     Image blur_img;
//     Size kernel_size = Size(3, 3);
//     double blur_sigma = 1.4;
//     image::process::gaussianBlur(gray_img, blur_img, kernel_size, blur_sigma);
//     Image canny_img;
//     int canny_threshold_1 = 50;
//     int canny_threshold_2 = 100;
//     image::process::cannyFilter(blur_img, canny_img, canny_threshold_1, canny_threshold_2);

//     std::vector<std::vector<image::Point>> contours;
//     image::process::findContours(canny_img, contours);


//     Image contour_img(canny_img.size());
//     std::vector<image::Color> colors_used;




//     std::mt19937 rng(std::random_device{}());
//     std::uniform_int_distribution<int> color_dist(64, 255);

//     for (std::vector<image::Point> segment : contours) { 
//         image::Color color;
//         do {
//             color = image::Color(color_dist(rng), color_dist(rng), color_dist(rng));
//         } while (std::find(colors_used.begin(), colors_used.end(), color) != colors_used.end());
        
//         colors_used.push_back(color);
        
//         for (const auto& pt : segment) {
//             if (pt.y >= 0 && pt.y < contour_img.getHeight() && pt.x >= 0 && pt.x < contour_img.getWidth()) {
//                 contour_img.setPixel(pt.x, pt.y, color);
//             }
//         }
//     }

//     io::saveImage(contour_img, "../Results/raw_contours.jpg");
// }

























// void ProcessTest::runAllTests() { 

//     cout << "RUNNING TESTS... " << endl;
//     printHorizontalBar();

//     chrono::duration<double> total_time(0.0);

//     // call test functions
//     total_time += timeFunction("Load Image", [&]() {loadImage();});
//     total_time += timeFunction("Squeeb Test", [&]() {squeebTest();});
//     total_time += timeFunction("Resize Image", [&]() {testResize();});
//     total_time += timeFunction("Convert To Gray", [&]() {testGrayscale();});
//     total_time += timeFunction("Blur Image", [&]() {testBlur();});
//     total_time += timeFunction("Sobel Filter", [&]() {testSobel();});
//     total_time += timeFunction("Canny Filter", [&]() {testCanny();});
//     total_time += timeFunction("Detect Contours", [&]() {testContours();});




//     printHorizontalBar();
//     printTotalTime(total_time);

//     cout << endl;




// }


// }