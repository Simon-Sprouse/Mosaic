#include "test.hpp"
#include "graphics.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <iomanip>
#include <random>



namespace Test { 

    void showDistanceVectorField(mosaic_gen::Mosaic& mosaic) { 

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
        mosaic.distance.convertTo(distance_visual, CV_8U, 255.0 / cv::norm(mosaic.distance, cv::NORM_INF));
        mosaic.saveImage(distance_visual, "distance_field");

        // tangent field


        int grid_size = 25;
        int max_step = 4;
        std::vector<cv::Point> grid_points = mosaic.samplePointsGrid(mosaic.segmented, grid_size);
        std::vector<cv::Point> samplePoints = mosaic.jitterPoints(grid_points, max_step, mosaic.segmented.size());
    
    
        std::vector<std::tuple<cv::Point, cv::Vec2f, float>> results;
    
        if (mosaic.segmented.empty() || samplePoints.empty()) {
            return;
        }
    
        
        // Step 5: Evaluate tangent and distance at each point
        for (const cv::Point& pt : samplePoints) {
            auto [tangent, dist] = mosaic.sampleTangentPoint(pt);
            results.emplace_back(pt, tangent, dist);
        }
    
        // Visualization
        cv::Mat vector_field = cv::Mat::zeros(mosaic.segmented.size(), CV_8UC3);
        const int length = 20;
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
            cv::Scalar color(bgr[0], bgr[1], bgr[2]);
    
            // Compute angle and draw arrow
            double angle_rad = std::atan2(tangent[1], tangent[0]);
            double angle_deg = angle_rad * 180.0 / CV_PI;
    
            Graphics::drawArrow(vector_field, pt, length, angle_deg, color);
        }
    
        mosaic.saveImage(vector_field, "vector_field");

        
    }


    void squeebTest() { 
        int n = 10000;
        for (int i = 0; i < n; i++) { 
            for (int j = 0; j < n; j++) { 
                int squeeb = 420;
            }
        }
    }


    void testFloodFillPoints(mosaic_gen::Mosaic& mosaic) { 

        // necessary progress
        mosaic.loadImage();

        mosaic.canvas = cv::Mat::zeros(mosaic.original.size(), CV_8UC3);


        int x = static_cast<int>(mosaic.canvas.cols / 2);
        int y = static_cast<int>(mosaic.canvas.rows / 2);
        cv::Point pt(x, y);

        double tile_size = 30;
        double theta_deg = 20;
        cv::Scalar color(255, 255, 255);
        Graphics::drawSquare(mosaic.canvas, pt, tile_size, theta_deg, color, 5);

        int num_points = 16;
        std::vector<cv::Point> next_points = mosaic.getFloodFillPoints2(pt, theta_deg, tile_size * 1.5, num_points);
        cv::Scalar point_color(255, 255, 0);
        for (cv::Point pt : next_points) { 
            Graphics::drawSquare(mosaic.canvas, pt, 5, theta_deg, point_color, 5);
          
        }

        mosaic.saveImage(mosaic.canvas, "flood_fill_test");

    }


    


    void testContours(mosaic_gen::Mosaic& mosaic) { 

        // necessary progress
        mosaic.loadImage();
        mosaic.resizeOriginal();
        mosaic.grayImage();
        mosaic.blurImage();
        mosaic.cannyFilter();
        mosaic.detectContours();
        mosaic.rankSegments();

        cv::Mat segment_canvas = cv::Mat::zeros(mosaic.edges.size(), CV_8UC3);
        std::vector<cv::Vec3b> colors_used;


        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> color_dist(64, 255);

        for (std::vector<cv::Point> segment : mosaic.segments) { 
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


    void testSegmentSelection(mosaic_gen::Mosaic& mosaic) { 

        // necessary progress
        mosaic.loadImage();
        mosaic.resizeOriginal();
        mosaic.grayImage();
        mosaic.blurImage();
        mosaic.cannyFilter();
        mosaic.detectContours();
        mosaic.rankSegments();

        

        int k = 2;
        mosaic.selectSegment(k);

        const cv::Mat img = mosaic.selected_segment;



        mosaic.saveImage(mosaic.selected_segment, "selected_segment");


    }



















    void printTestHeader(const std::string& test_name) {
        std::cout << std::left << std::setw(40) << ("[Running] " + test_name)
                  << " | ";
        std::cout.flush(); // flush in case timing starts immediately after
    }
    
    void printTestFooter(std::chrono::duration<double> elapsed) {
        std::cout << std::right << std::setw(12)
                  << std::fixed << std::setprecision(4)
                  << elapsed.count() << " s" << std::endl;
    }

    void printHorizontalBar() { 
        cout << std::string(41, '~') << " " << std::string(15, '~') << endl;
    }

    void printTotalTime(std::chrono::duration<double> total_time) { 


        cout << std::left << std::setw(40) << "[Total Time]"
        << std::right << std::setw(15) << std::fixed << std::setprecision(4)
        << total_time.count() << " s" << endl;
    }
    

    template <typename Func>
    chrono::duration<double> timeFunction(const std::string& name, Func&& fn) {
        printTestHeader(name);
        auto start = std::chrono::high_resolution_clock::now();
        fn(); // run the test
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        printTestFooter(elapsed);
        return elapsed;
    }


    void runAllTests(mosaic_gen::Mosaic& mosaic) { 

        cout << "RUNNING TESTS... " << endl;
        printHorizontalBar();

        chrono::duration<double> total_time(0.0);

        // call test functions
        total_time += timeFunction("Vector Field", [&]() {showDistanceVectorField(mosaic);});
        total_time += timeFunction("Squeeb Test", [&]() {squeebTest();});
        total_time += timeFunction("Flood Fill Points", [&]() {testFloodFillPoints(mosaic);});
        total_time += timeFunction("Test Contours", [&]() {testContours(mosaic);});
        total_time += timeFunction("Select Segment", [&]() {testSegmentSelection(mosaic);});


        printHorizontalBar();
        printTotalTime(total_time);

        cout << endl;




    }

    void runTimedProcess(mosaic_gen::Mosaic& mosaic) { 

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
        total_time += timeFunction("ranking contours", [&]() {mosaic.rankSegments();});
        total_time += timeFunction("place tiles along contours", [&]() {mosaic.placeTileAllSegments();});
        mosaic.saveImage(mosaic.mask, "mask_contours");

        total_time += timeFunction("place tiles with flood fill", [&]() {mosaic.showFloodFillPoints();});
        mosaic.saveImage(mosaic.mask, "mask_flood_fill");

        total_time += timeFunction("place tiles randomly", [&]() {mosaic.placeTileAllBackground();});
        mosaic.saveImage(mosaic.mask, "mask_random_fill");

        total_time += timeFunction("recontruct image", [&]() {mosaic.reconstructPlacedTiles();});
        mosaic.saveImage(mosaic.canvas,  "reconstruction");

        total_time += timeFunction("create gif from tile info", [&]() {mosaic.saveGif(20, "animation");});

        total_time += timeFunction("save tile info as csv", [&]() {mosaic.saveTileInfo("frontiers");});

        printHorizontalBar();
        printTotalTime(total_time);
        

        cout << endl;

    }

    void oldMain(mosaic_gen::Mosaic& mosaic) { 

        auto start = chrono::high_resolution_clock::now();

         // load image
        cout << "Loaded image: " << mosaic.image_name << endl;
        cout << "Original dimensions: " << mosaic.original.size() << endl;

        mosaic.loadImage();


        // Resize Image

        mosaic.resizeOriginal();
        mosaic.saveImage(mosaic.resized, "resized");
        cout << "Resized image to size: " << mosaic.resized.size() << endl;

        // Grayscale Image
        mosaic.grayImage();
        mosaic.saveImage(mosaic.grayscale, "gray");

        // Blur Image
        mosaic.blurImage();
        mosaic.saveImage(mosaic.blurred, "blurred");

        // Canny Filter
        mosaic.cannyFilter();
        mosaic.saveImage(mosaic.edges, "canny_edges");

        // Detect Contours
        int contour_count = mosaic.detectContours();
        mosaic.saveImage(mosaic.segmented, "segmented_edges");
        cout << "Detected: " << contour_count << " edges" << endl;


        // Rank Segments
        mosaic.rankSegments();

    
        // Place tiles on all segments
        mosaic.placeTileAllSegments();
        mosaic.saveImage(mosaic.mask, "mask");

        // show flood fill points
        mosaic.showFloodFillPoints();
        mosaic.saveImage(mosaic.canvas, "frontiers_canvas");
        mosaic.saveImage(mosaic.mask, "flood_fill_points");
        
    

        // sample points
        mosaic.placeTileAllBackground();
        mosaic.saveImage(mosaic.canvas, "samples");

        // store placed tiles and reconstruct
        mosaic.reconstructPlacedTiles();
        mosaic.saveImage(mosaic.canvas,  "reconstruction");
        


        // save gif
        mosaic.saveGif(20, "animation");

        // save csv tile placements
        mosaic.saveTileInfo("frontiers");
    


        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_time = end - start;
        cout << "Time to complete: " << elapsed_time.count() << " seconds" << endl;



        }

}