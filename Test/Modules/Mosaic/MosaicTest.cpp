#include "MosaicTest.hpp"

#include "../../../Code/Utils/Graphics/graphics.hpp"
#include "../../../Code/Utils/Image/io.hpp"
#include "../../../Code/Utils/Random/random.hpp"
#include "geometry.hpp"
#include "imageProcess.hpp"
#include <opencv2/imgproc.hpp>

namespace mosaic_gen::test { 

using image::Image;
using image::Size;
using image::Point;
using image::Color;
using image::io::saveImageFileSystem;




void MosaicTest::testConstructor() { 
    Mosaic mosaic(params_);
    // cout << "params_.tile_size: " << mosaic.params.tile_size << endl;
}


void MosaicTest::testPipeline() {

    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    saveImageFileSystem(mosaic.original, save_dir_ + "original.jpg");
    saveImageFileSystem(mosaic.resized, save_dir_ + "resized.jpg");


    Image strokes_img(mosaic.resized.size());
    Graphics::drawStrokesRandomColor(strokes_img, mosaic.strokes);

    saveImageFileSystem(strokes_img, save_dir_ + "strokes.jpg");


}

void MosaicTest::testSelectStroke() { 

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();

    std::vector<int> stroke_ids = {0, 1, 2, 10};

    for (int stroke_id : stroke_ids) {
        mosaic.selectStroke(stroke_id);
        saveImageFileSystem(mosaic.selected_stroke, save_dir_ + "selected_stroke" + std::to_string(stroke_id) + ".jpg");
    }
    
}


void MosaicTest::testMask() { 
    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    int size = mosaic.params.tile_size;

    int num_points = 500000;
    std::vector<Point> points = Random::randomPointsVector(mosaic.resized.size(), num_points);

    for (Point pt : points) { 
        double theta_deg = Random::randomDouble(0, 90);
        if (mosaic.isValidTile(pt, size, theta_deg)) {
            mosaic.placeTile(pt, size, theta_deg);
        }
    }

    saveImageFileSystem(mosaic.mask, save_dir_ + "mask.jpg");


}


// todo make parameters settable
void MosaicTest::testRandomStart() { 

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    mosaic.selectStroke(0);

    Image test_canvas = mosaic.selected_stroke.clone();
    Point pt = mosaic.getRandomPointOnStroke(0);
    int size = mosaic.params.tile_size;
    double theta_deg = 0;
    Color color(255, 0, 0);
    Graphics::drawSquare(test_canvas, pt, size, theta_deg, color, 2);

    saveImageFileSystem(test_canvas, save_dir_ + "first_tile.jpg");


}



void MosaicTest::testFindThetaStroke() { 


    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    mosaic.selectStroke(0);

    Image test_canvas = mosaic.selected_stroke.clone();
    Point pt = mosaic.getRandomPointOnStroke(0);
    int size = mosaic.params.tile_size;


    Color regionColor(0, 255, 0);
    std::vector<Point> region_pixels = mosaic.findNonZeroInRadius(mosaic.selected_stroke, pt, size);
    for (Point pt : region_pixels) { 
        test_canvas.setPixel(pt.x, pt.y, regionColor);
    }
    saveImageFileSystem(test_canvas, save_dir_ + "region_pixels.jpg");




    Vec2d direction = Geometry::pcaDirection(region_pixels);
    // cout << "direction from test: " << direction << endl;



    double theta_deg = mosaic.findBestTheta(pt, size);
    // cout << "theta deg: " << theta_deg << endl;
    Color color(255, 0, 0);
    Graphics::drawSquare(test_canvas, pt, size, theta_deg, color, 2);

    saveImageFileSystem(test_canvas, save_dir_ + "find_theta.jpg");


}






void MosaicTest::testRingIntersections() { 

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    mosaic.selectStroke(0);

    Image test_canvas = mosaic.selected_stroke.clone();
    Point pt = mosaic.getRandomPointOnStroke(0);
    int size = mosaic.params.tile_size;

    int ring_size = size * 2;
    double theta_deg = mosaic.findBestTheta(pt, size);
    int thickness = 2;
    std::vector<Point> intersections = mosaic.findRingIntersections(pt, ring_size, theta_deg, thickness);
    // saveImageFileSystem(mosaic.canvas, save_dir_ + "checked_points.jpg");


    // Draw ring
    Color ring_color(0, 0, 255);
    Graphics::drawSquare(test_canvas, pt, ring_size, theta_deg, ring_color, 2);

    // Draw center tile
    Color tile_color(255, 0, 0);
    Graphics::drawSquare(test_canvas, pt, size, theta_deg, tile_color, 2);

    // Draw intersections
    Color point_color(0, 255, 0);
    int point_size = 10;
    for (Point point : intersections) { 
        Graphics::drawSquare(test_canvas, point, point_size, 0, point_color, point_size);
    }
    
    saveImageFileSystem(test_canvas, save_dir_ + "intersections.jpg");


}


void MosaicTest::testMultipleRings() { 

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    mosaic.selectStroke(0);

    Image test_canvas = mosaic.selected_stroke.clone();
    Point pt = mosaic.getRandomPointOnStroke(0);
    int size = mosaic.params.tile_size;
    double theta_deg = mosaic.findBestTheta(pt, size);

    std::vector<Point> intersections = mosaic.findPointsMultipleRings(pt, theta_deg);


    // Draw center tile
    Color tile_color(255, 0, 0);
    Graphics::drawSquare(test_canvas, pt, size, theta_deg, tile_color, 2);

    // Draw intersections
    Color point_color(0, 255, 0);
    int point_size = 10;
    for (Point point : intersections) { 
        Graphics::drawSquare(test_canvas, point, point_size, 0, point_color, point_size);
    }
    
    saveImageFileSystem(test_canvas, save_dir_ + "multiple_ring_intersections.jpg");


}



void MosaicTest::testPlaceTileStroke() { 
    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline(); 
    mosaic.placeTilesAlongStroke(0);

    saveImageFileSystem(mosaic.mask, save_dir_ + "tiles_along_stroke0.jpg");
}




























void MosaicTest::testPlaceTileAllStrokes() { 

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline(); 
    mosaic.placeTilesAllStrokes();

    saveImageFileSystem(mosaic.mask, save_dir_ + "tiles_all_strokes.jpg");
}




void MosaicTest::testSquareBorderPoints() { 

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline(); 

    Image test_img(mosaic.resized.size());
    int distance_from_center = mosaic.params.distance_from_center;


    // Test once

    Size mosaic_size = mosaic.size();
    Point center = Point(mosaic_size.width / 2, mosaic_size.height / 2); // TODO overload this operator
    double theta_deg = 0.0;
    int size = mosaic.params.tile_size;
    int num_points = 8;
    std::vector<Point> border_points = Geometry::samplePointsSquareBorder(center, theta_deg, distance_from_center, num_points);


    Color tile_color(255);
    Graphics::drawSquare(test_img, center, size, theta_deg, tile_color, size);

    Color point_color(255, 0, 0);
    int point_size = 2;
    for (Point pt : border_points) { 
        Graphics::drawSquare(test_img, pt, point_size, theta_deg, point_color, point_size);
    }

    saveImageFileSystem(test_img, save_dir_ + "border_points.jpg");





    // Now test multiple

    test_img.fill(Color()); // TODO make reset function

    int grid_size = 30;
    std::vector<Point> center_points = Random::gridPointsVector(mosaic_size, grid_size); // TODO fix this to adjust top left positioning
    int num_border_points = 0;

    for (Point center : center_points) { 

        double theta_deg = 0.0;
        int size = mosaic.params.tile_size;
        std::vector<Point> border_points = Geometry::samplePointsSquareBorder(center, theta_deg, distance_from_center, num_border_points);


        Color tile_color(255);
        Graphics::drawSquare(test_img, center, size, theta_deg, tile_color, size);

        Color point_color = Random::randomColor();
        int point_size = 5;
        for (Point pt : border_points) { 
            Graphics::drawSquare(test_img, pt, point_size, theta_deg, point_color, point_size);
        }

        num_border_points++;
    }


    saveImageFileSystem(test_img, save_dir_ + "border_points_multiple.jpg");



}




void MosaicTest::testVectorField() { 

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    mosaic.computeDistanceField();

    int grid_size = 20;
    int max_step = 4;
    std::vector<Point> points = Random::gridPointsVector(mosaic.resized.size(), grid_size);
    std::vector<Point> jittered_points = Random::jitterPoints(points, max_step, mosaic.resized.size());

    Image test_canvas = mosaic.canny.clone();
    for (Point pt : jittered_points) { 
        double theta_deg = mosaic.findThetaTangent(pt);
        Color arrow_color(255, 0, 0);
        int length = 15;
        int thickness = 3;
        Graphics::drawArrow(test_canvas, pt, length, thickness, theta_deg, arrow_color);
    }

    saveImageFileSystem(test_canvas, save_dir_ + "vector_field.jpg");

}



void MosaicTest::testFloodFill() { 

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    mosaic.placeTilesAllStrokes();
    mosaic.floodFill();

    saveImageFileSystem(mosaic.mask, save_dir_ + "flood_fill.jpg");



}


void MosaicTest::testGapFill() { 
    

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    mosaic.placeTilesAllStrokes();
    mosaic.floodFill();
    mosaic.gapFill();

    saveImageFileSystem(mosaic.mask, save_dir_ + "gap_fill.jpg");


}



void MosaicTest::testReconstructImage() { 

    // necessary progress
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();
    mosaic.placeTilesAllStrokes();
    mosaic.floodFill();
    mosaic.gapFill();

    mosaic.reconstructImage();
 
    saveImageFileSystem(mosaic.canvas, save_dir_ + "reconstruction.jpg");

}


void MosaicTest::testStepOnce() { 


    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);

    // test first step
    mosaic.stepOnce();
    image::io::saveImageFileSystem(mosaic.mask, save_dir_ + "mask_first_step.jpg");


    // test all steps
    int num_steps = 0;
    while (mosaic.stepOnce()) {num_steps++;};
    cout << "num steps taken stepOnce() loop: " << num_steps << endl;
    
    mosaic.reconstructShowFrontiers();
    image::io::saveImageFileSystem(mosaic.canvas, save_dir_ + "all_steps.jpg");



}

void MosaicTest::testStepK() { 

    int k = 100000;
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.stepK(k);

    mosaic.reconstructImage();
    image::io::saveImageFileSystem(mosaic.canvas, save_dir_ + "step_k.png");

    cout << "length tiles_placed: " << mosaic.tiles_placed.size() << endl;

}





void MosaicTest::testCanny() { 

    int k = 100000;
    Mosaic mosaic(params_);
    Image original = image::io::loadImageFileSystem(image_path_);
    mosaic.loadExistingImage(original);
    mosaic.contourPipeline();


    image::io::saveImageFileSystem(mosaic.strokes_image, save_dir_ + "canny_test.png");




}






void MosaicTest::runAllTests() { 
    cout << "RUNNING TESTS... " << endl;
    timeFunctionBar();

    chrono::duration<double> total_time(0.0);

    // call test functions
    // total_time += timeFunction("Construct Mosaic", [&]() {testConstructor();});
    // total_time += timeFunction("Contour Pipeline", [&]() {testPipeline();});
    // total_time += timeFunction("Select Stroke", [&]() {testSelectStroke();});
    // total_time += timeFunction("Mask Overlap Check", [&]() {testMask();});
    // total_time += timeFunction("Choose Point on Stroke", [&]() {testRandomStart();});
    // total_time += timeFunction("Find Theta on Stroke", [&]() {testFindThetaStroke();});
    // total_time += timeFunction("Find Ring Intersections", [&]() {testRingIntersections();});
    // total_time += timeFunction("Multiple Ring Intersections", [&]() {testMultipleRings();});
    // total_time += timeFunction("Tiles Along Stroke", [&]() {testPlaceTileStroke();});


    // total_time += timeFunction("Tiles Along All Strokes", [&]() {testPlaceTileAllStrokes();});
    // total_time += timeFunction("Square Border Points", [&]() {testSquareBorderPoints();});
    // total_time += timeFunction("Show Vector Field", [&]() {testVectorField();});
    // total_time += timeFunction("Flood Fill", [&]() {testFloodFill();});
    // total_time += timeFunction("Gap Fill", [&]() {testGapFill();});
    // total_time += timeFunction("Reconstruct Image", [&]() {testReconstructImage();});


    // total_time += timeFunction("Step Once", [&]() {testStepOnce();});
    // total_time += timeFunction("Step k", [&]() {testStepK();});
    total_time += timeFunction("Test Canny", [&]() {testCanny();});
   



    timeFunctionBar();
    printTotalTime(total_time);

    cout << endl;
}












}