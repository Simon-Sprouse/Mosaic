#include "test.hpp"
#include "graphics.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>



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
        float gamma = 0.3; // to strecth color map; 
    
        float minDist = std::numeric_limits<float>::max();
        float maxDist = std::numeric_limits<float>::lowest();
        for (const auto& [_, __, dist] : results) {
            minDist = std::min(minDist, dist);
            maxDist = std::max(maxDist, dist);
        }
    
    
        for (const auto& [pt, tangent, dist] : results) {
            // std::cout << "Point: " << pt
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
            cv::applyColorMap(grayPixel, colorPixel, cv::COLORMAP_MAGMA);
    
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
        const int squeeb = 6;
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


    




    void printTestHeader(const std::string& test_name) { 
        cout << "Running Test: " << test_name;
    }

    void printTestFooter(chrono::duration<double> elapsed) { 
        cout << "   " << elapsed.count() << " seconds" << endl;
    }

    template <typename Func>
    void timeFunction(const std::string& name, Func&& fn) {
        printTestHeader(name);
        auto start = std::chrono::high_resolution_clock::now();
        fn(); // run the test
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        printTestFooter(elapsed);
    }


    void runAllTests(mosaic_gen::Mosaic& mosaic) { 

        cout << "Running tests... " << endl;


        timeFunction("Vector Field", [&]() {
            showDistanceVectorField(mosaic);
        });

        timeFunction("Squeeb Test", [&]() {
            squeebTest();
        });

        timeFunction("Flood Fill Points", [&]() {
            testFloodFillPoints(mosaic);
        });



        cout << endl;




    }

    

    void runTimedProcess(mosaic_gen::Mosaic& mosaic) { 

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