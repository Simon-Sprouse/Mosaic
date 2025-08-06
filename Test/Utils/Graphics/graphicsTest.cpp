#include "graphicsTest.hpp"

#include "../../../Code/Utils/Graphics/graphics.hpp"
#include "../../../Code/Utils/Image/io.hpp"
#include "../../../Code/Utils/Random/random.hpp"

namespace Graphics::test { 

using image::Image;
using image::Size;
using image::Point;
using image::io::saveImageFileSystem;



void GraphicsTest::testLine() { 

    Image img(400, 400);
    Point point_a = Random::randomPoint(img.size());
    Point point_b = Random::randomPoint(img.size());
    int thickness = 5;
    Color color = Random::randomColor();

    Graphics::drawLine(img, point_a, point_b, thickness, color);
    saveImageFileSystem(img, "../Results/line_test.jpg");

}


void GraphicsTest::testLineMultiple() { 

    Image img(400, 400);

    int num_points = 10;
    std::vector<Point> points = Random::randomPointsVector(img.size(), num_points);

    for (Point point_a : points) { 
        for (Point point_b : points) { 
           
            int thickness = 5;
            Color color = Random::randomColor();
            Graphics::drawLine(img, point_a, point_b, thickness, color);
        }
        
    }

    saveImageFileSystem(img, "../Results/draw_line_fc.jpg");




}



void GraphicsTest::testArrow() { 

    Image img(400, 400);
    Point center = Random::randomPoint(img.size());
    int length = 40;
    int thickness = 5;
    double theta_deg = Random::randomDouble(0, 90);
    Color color = Random::randomColor();

    Graphics::drawArrow(img, center, length, thickness, theta_deg, color);
    saveImageFileSystem(img, "../Results/arrow_test.jpg");



    int num_points = 210;
    img.fill(Color()); // reset 
    std::vector<Point> points = Random::randomPointsVector(img.size(), num_points);

    for (Point point : points) { 
       
           

        color = Random::randomColor();
        theta_deg = Random::randomDouble(0, 180);
        Graphics::drawArrow(img, point, length, thickness, theta_deg, color);
        
        
    }

    saveImageFileSystem(img, "../Results/arrow_test_multiple.jpg");


}






// TODO draw circles at the points
void GraphicsTest::testFillPolygon() {

    std::vector<int> num_points = {3, 6, 9, 12, 42};

    // TODO change random sample to take Size as a parameter, not image
    for (int number : num_points) { 

        Size size(400, 400);
        Image img(size);

        std::vector<Point> points = Random::randomPointsVector(img.size(), number);

        Color color(0, 255, 255); // TODO construct colors from hex codes
        Graphics::drawFilledPolygon(img, points, color);

        saveImageFileSystem(img, "../Results/polygon_test_" + std::to_string(number)  + "_points.jpg");

    }
    
}

void GraphicsTest::testDrawSquare() { 


    Image img(400, 400);
    Point center = Random::randomPoint(img.size());
    int size = 30;
    double theta_deg = Random::randomDouble(0, 90);
    Color color(255, 0, 0);
    int border_width = 5;


    Graphics::drawSquare(img, center, size, theta_deg, color, border_width);
    saveImageFileSystem(img, "../Results/draw_square_test.jpg");



}


void GraphicsTest::testDrawSquareMultiple() { 


    Image img(400, 400);

    int num_points = 420;
    std::vector<Point> points = Random::randomPointsVector(img.size(), num_points);

    for (Point point : points) { 

        int size = 30;
        double theta_deg = Random::randomDouble(0, 90);
        Color color = Random::randomColor();
        int border_width = 5;
    
        Graphics::drawSquare(img, point, size, theta_deg, color, border_width);
    }
    
   
    saveImageFileSystem(img, "../Results/draw_square_test_multiple.jpg");



}




void GraphicsTest::testRandom() { 

    Image img(400, 400);
    Color color = Random::randomColor();
    img.fill(color);
    saveImageFileSystem(img, "../Results/random_fill.jpg");

    // random points
    img.fill(Color());
    int num_points = 40;
    int size = 20;
    double theta_deg = 0;
    std::vector<Point> random_points = Random::randomPointsVector(img.size(), num_points);
    for (Point point : random_points) { 
        color = Random::randomColor();
        Graphics::drawSquare(img, point, size, theta_deg, color, size);
    }
    saveImageFileSystem(img, "../Results/random_points.jpg");

    // grid points
    img.fill(Color());
    int grid_size = 20;
    std::vector<Point> grid_points = Random::gridPointsVector(img.size(), grid_size);
    for (Point point : grid_points) { 
        color = Random::randomColor();
        Graphics::drawSquare(img, point, size, theta_deg, color, size);
    }
    saveImageFileSystem(img, "../Results/grid_points.jpg");

    // jitter grid
    img.fill(Color());
    int max_step = 2;
    std::vector<Point> jittered_points = Random::jitterPoints(grid_points, max_step, img.size());
    for (Point point : jittered_points) { 
        color = Random::randomColor();
        Graphics::drawSquare(img, point, size, theta_deg, color, size);
    }
    saveImageFileSystem(img, "../Results/jitter_points.jpg");


}










void GraphicsTest::runAllTests() { 
    cout << "RUNNING TESTS... " << endl;
    timeFunctionBar();

    chrono::duration<double> total_time(0.0);

    // call test functions
    total_time += timeFunction("Draw Line", [&]() {testLine();});
    total_time += timeFunction("Draw Line Multiple", [&]() {testLineMultiple();});
    total_time += timeFunction("Draw Arrow", [&]() {testArrow();});
    total_time += timeFunction("Fill Polygon", [&]() {testFillPolygon();});
    total_time += timeFunction("Draw Square", [&]() {testDrawSquare();});
    total_time += timeFunction("Draw Square Multiple", [&]() {testDrawSquareMultiple();});
    total_time += timeFunction("Test Random", [&]() {testRandom();});
   



    timeFunctionBar();
    printTotalTime(total_time);

    cout << endl;
}











}