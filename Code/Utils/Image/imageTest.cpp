#include "imageTest.hpp"

#include <iostream>
#include <sstream>
#include <cassert>

using namespace std;
using std::ostringstream;
using image::Image, image::Color, image::Size, image::Point;


namespace image::test { 



bool ImageTest::testStructPrint() {

    bool all_passed = true;

    all_passed &= checkPrint("Size default", Size(), "size[0 x 0]");
    all_passed &= checkPrint("Size(int, int)", Size(5, 10), "size[5 x 10]");
    all_passed &= checkPrint("Point default", Point(), "point[0, 0]");
    all_passed &= checkPrint("Point(int, int)", Point(3, 4), "point[3, 4]");
    all_passed &= checkPrint("Point(float, float)", Point(1.5f, 2.5f), "point[1, 2]");
    all_passed &= checkPrint("Point(float, float)", Point(7.9, 8.1), "point[7, 8]");
    all_passed &= checkPrint("Color default", Color(), "rgba[0, 0, 0, 255]");
    all_passed &= checkPrint("Color(uint8_t)", Color(12), "rgba[12, 12, 12, 255]");
    all_passed &= checkPrint("Color(int, int, int)", Color(10, 20, 30), "rgba[10, 20, 30, 255]");
    all_passed &= checkPrint("Color(int, int, int, int)", Color(10, 20, 30, 40), "rgba[10, 20, 30, 40]");
    all_passed &= checkPrint("Color bounds check", Color(-10, 999, 128, 256), "rgba[0, 255, 128, 255]");
    all_passed &= checkPrint("Color(float)", Color(44.5f), "rgba[42, 42, 42, 255]");
    all_passed &= checkPrint("Color(float, float, float)", Color(1.0f, 2.0f, 3.0f), "rgba[1, 2, 3, 255]");
    all_passed &= checkPrint("Color(float, float, float, float)", Color(4.5f, 5.5f, 6.5f, 7.5f), "rgba[4, 5, 6, 7]");
    all_passed &= checkPrint("Color(float, float, float, float)", Color(254.9, 253.1, 252.0, 251.8), "rgba[254, 253, 252, 251]");

    return all_passed;
}



bool ImageTest::testImageRuleOf5() {

    bool all_passed = true;

    // Default constructor
    all_passed &= checkEqual("Default Constructor size", Image(), Image(0, 0));

    // Constructor (int, int)
    Image img_w_h(3, 4);
    all_passed &= checkEqual("Constructor (w,h) size", img_w_h, Image(Size(3, 4)));
    all_passed &= checkEqual("Constructor (w,h) default fill", img_w_h.at(0, 0), Color());

    // Constructor (int, int, Color)
    Image img_w_h_fill(2, 2, Color(10, 20, 30, 255));
    all_passed &= checkEqual("Constructor (w,h,color) size", img_w_h_fill.size(), Size(2, 2));
    all_passed &= checkEqual("Constructor (w,h,color) fill check", img_w_h_fill.at(1, 1), Color(10, 20, 30, 255));

    // Constructor (Size)
    Size s1(5, 5);
    Image img_size(s1);
    all_passed &= checkEqual("Constructor (Size) size", img_size.size(), s1);
    all_passed &= checkEqual("Constructor (Size) default fill", img_size.at(4, 4), Color());

    // Constructor (Size, Color)
    Size s2(2, 3);
    Color blueish(10, 50, 200);
    Image img_size_fill(s2, blueish);
    all_passed &= checkEqual("Constructor (Size, Color) size", img_size_fill.size(), s2);
    all_passed &= checkEqual("Constructor (Size, Color) fill", img_size_fill.at(1, 2), blueish);

    // Copy constructor
    Image img2(img_w_h_fill);
    all_passed &= checkEqual("Copy Constructor equality", img2, img_w_h_fill);

    // Modify original and ensure copy is unchanged
    img_w_h_fill.at(0, 0) = Color(1, 2, 3, 255);
    all_passed &= checkEqual("Copy unaffected by original mutation", img2.at(0, 0), Color(10, 20, 30, 255));

    // Copy assignment
    Image img3;
    img3 = img_w_h_fill;
    all_passed &= checkEqual("Copy assignment equality", img3, img_w_h_fill);

    // Move constructor
    Image img4(std::move(img3));
    all_passed &= checkEqual("Move Constructor result", img4.size(), Size(2, 2));
    all_passed &= checkEqual("Move Constructor source cleared", img3.size(), Size(0, 0));

    // Move assignment
    Image img5;
    img5 = std::move(img4);
    all_passed &= checkEqual("Move assign result", img5.size(), Size(2, 2));
    all_passed &= checkEqual("Move assign source cleared", img4.size(), Size(0, 0));

    return all_passed;
}






bool ImageTest::testImagePrint() {
   
    bool all_passed = true;

    all_passed &= checkPrint("Image default", Image(), "Image[0 x 0]");
    all_passed &= checkPrint("Image(Size, Color)", Image(Size(5, 7), Color(0, 0, 0)), "Image[5 x 7]");
    all_passed &= checkPrint("Image(w, h, Color)", Image(3, 9, Color(255, 255, 255)), "Image[3 x 9]");

    return all_passed;
}




void ImageTest::runAllTests() { 
    runFunction("Structs Print", [&]() {return testStructPrint();});
    runFunction("Image Rule of 5", [&]() {return testImageRuleOf5();});
    runFunction("Image Print", [&]() {return testImagePrint();});
}






}



