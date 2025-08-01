#include "test.hpp"
#include "graphics.hpp"
#include "random.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include <random>



namespace mosaic_gen::Test { 

MosaicTest::MosaicTest(mosaic_gen::Mosaic& mosaic) : mosaic(mosaic) {}


void MosaicTest::showDistanceVectorField() { 

    // necessary progress
    mosaic.loadImage();
    mosaic.resizeOriginal();
    mosaic.grayImage();
    mosaic.blurImage();
    mosaic.cannyFilter();
    mosaic.detectContours();

    // vector field test begins here

    // distance field
    mosaic.computeDistanceField();
    cv::Mat distance_visual;
    mosaic.distance_map.convertTo(distance_visual, CV_8U, 255.0 / cv::norm(mosaic.distance_map, cv::NORM_INF));
    mosaic.saveImage(distance_visual, "distance_field");

    // tangent field


    int grid_size = static_cast<int>(mosaic.params.tile_size * 1.5);
    int max_step = static_cast<int>(mosaic.params.tile_size * 0.5);
    std::vector<cv::Point> grid_points = Random::samplePointsGrid(mosaic.contours, grid_size);
    std::vector<cv::Point> samplePoints = Random::jitterPoints(grid_points, max_step, mosaic.contours.size());


    std::vector<std::tuple<cv::Point, cv::Vec2f, float>> results;

    if (mosaic.contours.empty() || samplePoints.empty()) {
        return;
    }

    
    // Step 5: Evaluate tangent and distance at each point
    for (const cv::Point& pt : samplePoints) {
        auto [tangent, dist] = mosaic.getTangentAtPoint(pt);
        results.emplace_back(pt, tangent, dist);
    }

    // Visualization
    cv::Mat vector_field = cv::Mat::zeros(mosaic.contours.size(), CV_8UC3);
    const int length = mosaic.params.tile_size;
    float gamma = 0.6; // to strecth color map; 

    float minDist = std::numeric_limits<float>::max();
    float maxDist = std::numeric_limits<float>::lowest();
    for (const auto& [_, __, dist] : results) {
        minDist = std::min(minDist, dist);
        maxDist = std::max(maxDist, dist);
    }


    for (const auto& [pt, tangent, dist] : results) {
        // cout << "Point: " << pt
        //         << " | Tangent: (" << tangent[0] << ", " << tangent[1] << ")"
        //         << " | Distance: " << dist << "\n";

        // Normalize distance to [0, 255]
        int value = 0;
        if (maxDist > minDist) {
            float normalized = (dist - minDist) / (maxDist - minDist);
            value = static_cast<int>(255.0f * std::pow(normalized, gamma));
            value = std::clamp(value, 0, 255);

        }

        // Create 1-pixel grayscale image
        cv::Mat grayPixel(1, 1, CV_8U, cv::Scalar(value));
        cv::Mat colorPixel;
        cv::applyColorMap(grayPixel, colorPixel, cv::COLORMAP_INFERNO);

        // Extract BGR color from the pixel
        cv::Vec3b bgr = colorPixel.at<cv::Vec3b>(0, 0);


        // Compute angle and draw arrow
        double angle_rad = std::atan2(tangent[1], tangent[0]);
        double angle_deg = angle_rad * 180.0 / CV_PI;

        int thickness = 2;
        Graphics::drawArrow(vector_field, pt, length, thickness, angle_deg, bgr);
    }

    mosaic.saveImage(vector_field, "vector_field");

    
}


void MosaicTest::squeebTest() { 
    int n = 10000;
    for (int i = 0; i < n; i++) { 
        for (int j = 0; j < n; j++) { 
            int squeeb = 420;
        }
    }
}


void MosaicTest::testFloodFillPoints() { 

    // necessary progress
    mosaic.loadImage();

    mosaic.canvas = cv::Mat::zeros(mosaic.original.size(), CV_8UC3);


    int x = static_cast<int>(mosaic.canvas.cols / 2);
    int y = static_cast<int>(mosaic.canvas.rows / 2);
    cv::Point pt(x, y);

    double tile_size = 30;
    double theta_deg = 20;
    cv::Vec3b color(255, 255, 255);
    Graphics::drawSquare(mosaic.canvas, pt, tile_size, theta_deg, color, 5);

    int num_points = 16;
    std::vector<cv::Point> next_points = mosaic.nextFrontierFromTile(pt, theta_deg, tile_size * 1.5, num_points);
    cv::Vec3b point_color(255, 255, 0);
    for (cv::Point pt : next_points) { 
        Graphics::drawSquare(mosaic.canvas, pt, 5, theta_deg, point_color, 5);
        
    }

    mosaic.saveImage(mosaic.canvas, "flood_fill_test");

}


void MosaicTest::testFloodFillFrontier() { 


    // necessary progress
    mosaic.loadImage();
    mosaic.resizeOriginal();
    mosaic.grayImage();
    mosaic.blurImage();
    mosaic.cannyFilter();
    mosaic.detectContours();
    mosaic.rankContours();
    mosaic.placeTileAllContours();
    



    cv::Mat flood_fill_canvas = mosaic.contours.clone();

    

    double point_size = 4;
    cv::Vec3b tile_color(255, 255, 255);
    cv::Vec3b point_color(0, 0, 255);

    int frontier = 0;

    for (const mosaic_gen::TileInfo& tile : mosaic.tiles_placed) {
        int num_points = mosaic.params.flood_fill_neighbor_points;
        int max_step = mosaic.getJitter(frontier);
        std::vector<cv::Point> points = mosaic.nextFrontierFromTile(tile.center, tile.theta_deg, mosaic.params.distance_from_center, num_points);
        std::vector<cv::Point> jittered_points = Random::jitterPoints(points, max_step, mosaic.mask.size());
        


        // print tile
        int border_width = static_cast<int>(tile.size * 0.25);
        Graphics::drawSquare(flood_fill_canvas, tile.center, tile.size, tile.theta_deg, tile_color, border_width);

        // print next points
        for (cv::Point point : jittered_points) { 
            Graphics::drawSquare(flood_fill_canvas, point, point_size, tile.theta_deg, point_color, point_size);
        }

    }


    
    mosaic.saveImage(flood_fill_canvas, "flood_fill_next_points");
    mosaic.resetData();



}





void MosaicTest::testContours() { 

    // necessary progress
    mosaic.loadImage();
    mosaic.resizeOriginal();
    mosaic.grayImage();
    mosaic.blurImage();
    mosaic.cannyFilter();
    mosaic.detectContours();
    mosaic.rankContours();

    cv::Mat segment_canvas = cv::Mat::zeros(mosaic.edges.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors_used;


    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> color_dist(64, 255);

    for (std::vector<cv::Point> segment : mosaic.segment_points) { 
        cv::Vec3b color;
        do {
            color = cv::Vec3b(color_dist(rng), color_dist(rng), color_dist(rng));
        } while (std::find(colors_used.begin(), colors_used.end(), color) != colors_used.end());
        
        colors_used.push_back(color);
        
        for (const auto& pt : segment) {
            if (pt.y >= 0 && pt.y < segment_canvas.rows && pt.x >= 0 && pt.x < segment_canvas.cols) {
                segment_canvas.at<cv::Vec3b>(pt.y, pt.x) = color;
            }
        }
    }

    mosaic.saveImage(segment_canvas, "segment_canvas");

    

}


void MosaicTest::testSegmentSelection() { 

    // necessary progress
    mosaic.loadImage();
    mosaic.resizeOriginal();
    mosaic.grayImage();
    mosaic.blurImage();
    mosaic.cannyFilter();
    mosaic.detectContours();
    mosaic.rankContours();

    

    int k = 2;
    mosaic.selectContour(k);

    const cv::Mat img = mosaic.selected_contour;



    mosaic.saveImage(mosaic.selected_contour, "selected_segment");


}


void MosaicTest::testSegmentOrder() { 

    // necessary progress
    mosaic.loadImage();
    mosaic.resizeOriginal();
    mosaic.grayImage();
    mosaic.blurImage();
    mosaic.cannyFilter();
    mosaic.detectContours();
    mosaic.rankContours();

    

    int k = 0;
    mosaic.selectContour(k);

    cv::Mat segment_canvas = mosaic.selected_contour.clone();

    mosaic.placeTileContour(k);

    cv::Vec3b tile_color(255, 255, 0);

    for (mosaic_gen::TileInfo tile : mosaic.tiles_placed) { 
        Graphics::drawSquareText(segment_canvas, tile.center, tile.size, tile.theta_deg, tile_color, tile.size, std::to_string(tile.order));
    }
    

    mosaic.saveImage(segment_canvas, "segment_order");
    mosaic.resetData();

}





void MosaicTest::testIntersections() { 



    // necessary progress
    mosaic.loadImage();
    mosaic.resizeOriginal();
    mosaic.grayImage();
    mosaic.blurImage();
    mosaic.cannyFilter();
    mosaic.detectContours();
    mosaic.rankContours();
    int k = 0;
    mosaic.selectContour(k);

    cv::Mat intersection_canvas = mosaic.selected_contour.clone();
    double point_size = 4;
    cv::Vec3b center_color(0, 0, 255);
    cv::Vec3b point_color(255, 255, 0);
    cv::Vec3b ring_color(0, 255, 0);

    



    cv::Point point = mosaic.getRandomPointOnContour(k);
    



    double initial_size = mosaic.params.tile_size;
    double theta_deg = mosaic.findBestTheta(point, initial_size);
    Graphics::drawSquare(intersection_canvas, point, point_size, theta_deg, center_color, point_size);

    std::vector<cv::Point> allIntersections;
    int radius = static_cast<int>(initial_size * 5);
    

    for (int i = 0; i < mosaic.params.number_of_rings; ++i) {


        double currentSize = initial_size + i * mosaic.params.step_size;

        std::vector<cv::Point> ringIntersections = mosaic.findRingIntersections(
            mosaic.selected_contour, point, currentSize, theta_deg
        );

        // draw ring
        Graphics::drawSquare(intersection_canvas, point, currentSize, theta_deg, ring_color, 1);

        allIntersections.insert(allIntersections.end(), ringIntersections.begin(), ringIntersections.end());
    }




    allIntersections = mosaic.filterUniqueIntersections(allIntersections);
    for (cv::Point intersection : allIntersections) {
        Graphics::drawSquare(intersection_canvas, intersection, point_size, theta_deg, point_color, point_size);
    }

    mosaic.saveImage(intersection_canvas, "test_intersections");



}












void MosaicTest::visualizePlacementMethod() { 

    // necessary progress
    if (mosaic.mask.empty()) { 
        mosaic.loadImage();
        mosaic.resizeOriginal();
        mosaic.grayImage();
        mosaic.blurImage();
        mosaic.cannyFilter();
        mosaic.detectContours();
        mosaic.rankContours();
        mosaic.placeTileAllContours();
        mosaic.floodFill();
        mosaic.gapFill();
    }


    cv::Vec3b contour_color(0, 0, 255);
    cv::Vec3b flood_fill_color(0, 255, 0);
    cv::Vec3b gap_fill_color(255, 0, 0);


    cv::Mat frontier_canvas = cv::Mat::zeros(mosaic.edges.size(), CV_8UC3);

    for (mosaic_gen::TileInfo tile : mosaic.tiles_placed) { 
        if (tile.frontier == 0) { 
            Graphics::drawSquare(frontier_canvas, tile.center, tile.size, tile.theta_deg, contour_color, tile.size);
        }
        else if (tile.frontier > 0) { 
            Graphics::drawSquare(frontier_canvas, tile.center, tile.size, tile.theta_deg, flood_fill_color, tile.size);
        }
        else { 
            Graphics::drawSquare(frontier_canvas, tile.center, tile.size, tile.theta_deg, gap_fill_color, tile.size);
        }
    }

    mosaic.saveImage(frontier_canvas, "placement_canvas");




}



void MosaicTest::visualizePlacementOrder() { 

    // necessary progress
    if (mosaic.mask.empty()) { 
        mosaic.loadImage();
        mosaic.resizeOriginal();
        mosaic.grayImage();
        mosaic.blurImage();
        mosaic.cannyFilter();
        mosaic.detectContours();
        mosaic.rankContours();
        mosaic.placeTileAllContours();
        mosaic.floodFill();
        mosaic.gapFill();
    }
    

    // Visualization
    cv::Mat ordering_canvas = cv::Mat::zeros(mosaic.contours.size(), CV_8UC3);
    const int length = mosaic.tiles_placed.size();
    float gamma = 0.6; // to strecth color map; 

    float minDist = 0;
    float maxDist = length;



    for (mosaic_gen::TileInfo tile : mosaic.tiles_placed) {

        int value = 0;
        if (maxDist > minDist) {
            float normalized = (tile.order - minDist) / (maxDist - minDist);
            value = static_cast<int>(255.0f * std::pow(normalized, gamma));
            value = std::clamp(value, 0, 255);

        }

        // Create 1-pixel grayscale image
        cv::Mat grayPixel(1, 1, CV_8U, cv::Scalar(value));
        cv::Mat colorPixel;
        cv::applyColorMap(grayPixel, colorPixel, cv::COLORMAP_CIVIDIS);

        // Extract BGR color from the pixel
        cv::Vec3b bgr = colorPixel.at<cv::Vec3b>(0, 0);



        Graphics::drawSquare(ordering_canvas, tile.center, tile.size, tile.theta_deg, bgr, tile.size);
    }

    mosaic.saveImage(ordering_canvas, "ordering_canvas");


}



void MosaicTest::visualizeFrontierOrder() { 

    // necessary progress
    if (mosaic.mask.empty()) { 
        mosaic.loadImage();
        mosaic.resizeOriginal();
        mosaic.grayImage();
        mosaic.blurImage();
        mosaic.cannyFilter();
        mosaic.detectContours();
        mosaic.rankContours();
        mosaic.placeTileAllContours();
        mosaic.floodFill();
        mosaic.gapFill();
    }
    

    // Visualization
    cv::Mat ordering_canvas = cv::Mat::zeros(mosaic.contours.size(), CV_8UC3);
    const int length = mosaic.tiles_placed.size();
    float gamma = 0.6; // to strecth color map; 



    int max_frontier = 0;
    for (mosaic_gen::TileInfo tile : mosaic.tiles_placed) { 
        if (tile.frontier >= max_frontier) { 
            max_frontier = tile.frontier;
        }
    }

    float minDist = 0;
    float maxDist = max_frontier;



    for (mosaic_gen::TileInfo tile : mosaic.tiles_placed) {

        int current_frontier = tile.frontier;
        if (tile.frontier < 0) { 
            current_frontier = max_frontier + 1;
        }

        int value = 0;
        if (maxDist > minDist) {
            float normalized = (current_frontier - minDist) / (maxDist - minDist);
            value = static_cast<int>(255.0f * std::pow(normalized, gamma));
            value = std::clamp(value, 0, 255);

        }

        // Create 1-pixel grayscale image
        cv::Mat grayPixel(1, 1, CV_8U, cv::Scalar(value));
        cv::Mat colorPixel;
        cv::applyColorMap(grayPixel, colorPixel, cv::COLORMAP_INFERNO);

        // Extract BGR color from the pixel
        cv::Vec3b bgr = colorPixel.at<cv::Vec3b>(0, 0);



        Graphics::drawSquare(ordering_canvas, tile.center, tile.size, tile.theta_deg, bgr, tile.size);
    }

    mosaic.saveImage(ordering_canvas, "frontier_ordering_canvas");


}

















void MosaicTest::printTestHeader(const std::string& test_name) {
    std::cout << std::left << std::setw(40) << ("[Running] " + test_name)
                << " | ";
    std::cout.flush(); // flush in case timing starts immediately after
}

void MosaicTest::printTestFooter(std::chrono::duration<double> elapsed) {
    std::cout << std::right << std::setw(12)
                << std::fixed << std::setprecision(4)
                << elapsed.count() << " s" << std::endl;
}

void MosaicTest::printHorizontalBar() { 
    cout << std::string(41, '~') << " " << std::string(15, '~') << endl;
}

void MosaicTest::printTotalTime(std::chrono::duration<double> total_time) { 

    cout << std::left << std::setw(40) << "[Total Time]"
    << std::right << std::setw(15) << std::fixed << std::setprecision(4)
    << total_time.count() << " s" << endl;
}





void MosaicTest::runAllTests() { 

    cout << "RUNNING TESTS... " << endl;
    printHorizontalBar();

    chrono::duration<double> total_time(0.0);

    // call test functions
    total_time += timeFunction("Vector Field", [&]() {showDistanceVectorField();});
    total_time += timeFunction("Squeeb Test", [&]() {squeebTest();});
    total_time += timeFunction("Test Contours", [&]() {testContours();});
    total_time += timeFunction("Select Segment", [&]() {testSegmentSelection();});
    total_time += timeFunction("DFS Segment Order", [&]() {testSegmentOrder();});
    total_time += timeFunction("Test Intersections", [&]() {testIntersections();});
    total_time += timeFunction("Flood Fill Points", [&]() {testFloodFillPoints();});
    total_time += timeFunction("Flood Fill Segment", [&]() {testFloodFillFrontier();});
    total_time += timeFunction("Visualize Placement", [&]() {visualizePlacementMethod();});
    total_time += timeFunction("Visualize Order", [&]() {visualizePlacementOrder();});
    total_time += timeFunction("Visualize Frontier", [&]() {visualizeFrontierOrder();});



    printHorizontalBar();
    printTotalTime(total_time);

    cout << endl;




}

void MosaicTest::runTimedProcess() { 

    // clean start
    mosaic.resetData();

    cout << "RUNNING PROCESS... " << endl;
    printHorizontalBar();

    chrono::duration<double> total_time(0.0);

    total_time += timeFunction("loading image", [&]() {mosaic.loadImage();});
    total_time += timeFunction("resize image", [&]() {mosaic.resizeOriginal();});
    mosaic.saveImage(mosaic.resized, "resized_original");

    total_time += timeFunction("convert image to grayscale", [&]() {mosaic.grayImage();});
    mosaic.saveImage(mosaic.grayscale, "grayscale");

    total_time += timeFunction("blur image", [&]() {mosaic.blurImage();});
    mosaic.saveImage(mosaic.blurred, "blurred");

    total_time += timeFunction("apply canny filter", [&]() {mosaic.cannyFilter();});
    mosaic.saveImage(mosaic.edges, "canny_edges");

    total_time += timeFunction("detecting contours", [&]() {mosaic.detectContours();});
    total_time += timeFunction("ranking contours", [&]() {mosaic.rankContours();});
    total_time += timeFunction("place tiles along contours", [&]() {mosaic.placeTileAllContours();});
    mosaic.saveImage(mosaic.mask, "mask_contours");

    total_time += timeFunction("place tiles with flood fill", [&]() {mosaic.floodFill();});
    total_time += timeFunction("place tiles with gap fill", [&]() {mosaic.gapFill();});
    mosaic.saveImage(mosaic.mask, "mask_random_fill");

    total_time += timeFunction("recontruct image", [&]() {mosaic.reconstructImage();});
    mosaic.saveImage(mosaic.canvas,  "reconstruction");

    total_time += timeFunction("create gif from tile info", [&]() {mosaic.saveGif(20, "animation");});

    total_time += timeFunction("save tile info as csv", [&]() {mosaic.saveTileInfo("frontiers");});

    printHorizontalBar();
    printTotalTime(total_time);
    

    cout << endl;

}

}