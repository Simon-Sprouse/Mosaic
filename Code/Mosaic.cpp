#include "Mosaic.hpp"
#include "graphics.hpp"
#include "geometry.hpp"
#include "random.hpp"

#include "gif.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stack>
#include <sstream>
#include <fstream>
#include <queue>

using namespace std;
namespace fs = std::__fs::filesystem;

namespace mosaic_gen {

// param constructor
Mosaic::Mosaic(const Parameters& hp) { 
    params = hp;

    loadImage();
    resizeOriginal();
    cout << "Loaded image: " << image_name << endl;
    cout << "Original dimensions: " << original.size() << endl;
    cout << "Resized image to size: " << resized.size() << endl;
    cout << endl;



    
}

void Mosaic::setWindow(std::string name) { 
    window_name = name;
}

void Mosaic::resetData() {


    original.release();
    resized.release();
    grayscale.release();
    blurred.release();
    edges.release();
    contours.release();
    distance_map.release();
    gradX.release();
    gradY.release();

    selected_contour.release();
    canvas.release();
    mask.release();

    segment_points.clear();
    segment_lengths.clear();

    tiles_placed.clear();
}


void Mosaic::runAll() {
    loadImage();
    resizeOriginal();
    grayImage();
    blurImage();
    cannyFilter();
    detectContours();
    rankContours();
    placeTileAllContours();
    floodFill();
    gapFill();
    renderTiles();
}

cv::Mat Mosaic::getCanvas() { 
    return canvas.clone();
}
















void Mosaic::saveImage(const cv::Mat& image,  const std::string& suffix) { 

    string output_dir = params.results_dir;


    if (image.empty()) { 
        return;
    }

    if (!fs::exists(output_dir)) { 
        fs::create_directory(output_dir);
    }

    std::string output_path = output_dir + "/" + image_name + "_" + suffix + ".jpg";
    if (cv::imwrite(output_path, image)) { 
        // cout << "Saved: " << output_path << endl;
    }
    else { 
        cerr << "Failed to save: " << output_path << endl;
    }


}


void Mosaic::saveGif(int tilesPerFrame, const std::string& suffix) {

    string output_dir = params.results_dir;


    int width = canvas.cols;
    int height = canvas.rows;

    std::string gifFilename = output_dir + "/" + image_name + "_" + suffix + ".gif";

    GifWriter writer;

    GifBegin(&writer, gifFilename.c_str(), width, height, 10); // delay in 1/100s

    cv::Mat gifCanvas = cv::Mat::zeros(canvas.size(), CV_8UC3);
    for (size_t i = 0; i < tiles_placed.size(); i += tilesPerFrame) {
        for (size_t j = i; j < std::min(i + tilesPerFrame, tiles_placed.size()); ++j) {
            const TileInfo& tile = tiles_placed[j];
            cv::Vec3b color = sampleTileColor(tile);
            Graphics::drawSquare(gifCanvas, tile.center, tile.size, tile.theta_deg, color, tile.size);
        }

        // Convert to RGBA
        cv::Mat rgba;
        cv::cvtColor(gifCanvas, rgba, cv::COLOR_BGR2RGBA);

        GifWriteFrame(&writer, rgba.data, width, height, 10);
    }


    // Hold on final frame by repeating it

    int final_hold_frames = 10;

    cv::Mat rgbaFinal;
    cv::cvtColor(gifCanvas, rgbaFinal, cv::COLOR_BGR2RGBA);
    for (int k = 0; k < final_hold_frames; ++k) {
        GifWriteFrame(&writer, rgbaFinal.data, width, height, 10);
    }

    GifEnd(&writer);
    // std::cout << "Saved animated GIF to: " << gifFilename << std::endl;
}




void Mosaic::saveTileInfo(const std::string& suffix) { 

    string output_dir = params.results_dir;


    std::ostringstream oss;

    // Write the CSV header
    oss << "center_x,center_y,size,theta_deg,order,frontier\n";

    // Iterate through each TileInfo struct in the vector
    for (const auto& tile : tiles_placed) {
        oss << tile.center.x << ","
            << tile.center.y << ","
            << tile.size << ","
            << tile.theta_deg << ","
            << tile.order << ","
            << tile.frontier << "\n";
    }

    std::string fileName = output_dir + "/" + image_name + "_" + suffix + ".csv";
    std::ofstream outFile(fileName); // Open the file for writing
    if (outFile.is_open()) {
        outFile << oss.str(); // Write the CSV content to the file
        outFile.close();      // Close the file
        // std::cout << "CSV data successfully written to " << fileName << std::endl;
    } else {
        std::cerr << "Error: Unable to open file '" << fileName << "' for writing." << std::endl;
    }



}







int Mosaic::getJitter(int frontier) {
    if (params.jitter_map.empty()) { 
        return 0;
    }
    for (const auto& [threshold, jitter] : params.jitter_map) {
        if (frontier < threshold) {
            return jitter;
        }
    }
    return 0;
}























void Mosaic::loadImage() { 
    string image_path = params.image_path;
    original = cv::imread(image_path);

    if (original.empty()) { 
        cerr << "Error: Could not load image from path: " << image_path << endl;
        return;
    }

    image_name = fs::path(params.image_path).stem().string();

}


void Mosaic::resizeOriginal() { 
    if (original.empty()) { 
        cerr << "Resized called but no original image found" << endl;
        return;
    }

    cv::resize(original, resized, cv::Size(), params.resize_factor, params.resize_factor, cv::INTER_LINEAR);

}


void Mosaic::grayImage() { 
    if (resized.empty()) { 
        cerr << "Gray called but no resized image" << endl;
        return;
    }
    cv::cvtColor(resized, grayscale, cv::COLOR_BGR2GRAY);
}


void Mosaic::blurImage() { 
    if (grayscale.empty()) { 
        cerr << "Blur called but no grayscale image" << endl;
        return;
    }

    // ensure odd kernel size
    if (params.blur_kernel_size % 2 == 0) { 
        params.blur_kernel_size += 1;
    }

    cv::GaussianBlur(grayscale, blurred, cv::Size(params.blur_kernel_size, params.blur_kernel_size), params.blur_sigma);
}



void Mosaic::cannyFilter() { 
    if (blurred.empty()) { 
        cerr << "Canny called but no blurred" << endl;
        return;
    }
    cv::Canny(blurred, edges, params.canny_threshold_1, params.canny_threshold_2);
}

// TODO move image processing pipeline to new class
int Mosaic::detectContours() { 
    if (edges.empty()) {
        cerr << "DetectContours called but no edges" << endl;
        return -1;
    }

    // Find contours
    std::vector<std::vector<cv::Point>> cv_contours;
    cv::findContours(edges.clone(), cv_contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Create an output color image
    contours = cv::Mat::zeros(edges.size(), CV_8UC3);
    int contour_id = 0;

 

    for (const std::vector<cv::Point>& contour : cv_contours) {
        if (contour.size() < 3)
            continue;

        std::vector<int> breaks;
        int len = contour.size();
        int w = params.segment_angle_window;

        for (int i = w; i < len - w; ++i) {
            cv::Point2f v1 = contour[i] - contour[i - w];
            cv::Point2f v2 = contour[i + w] - contour[i];

            double norm1 = std::sqrt(v1.x * v1.x + v1.y * v1.y) + 1e-8;
            double norm2 = std::sqrt(v2.x * v2.x + v2.y * v2.y) + 1e-8;

            cv::Point2f n1 = v1 / norm1;
            cv::Point2f n2 = v2 / norm2;

            double cosine = std::clamp(n1.dot(n2), -1.0f, 1.0f);
            double angle = std::acos(std::abs(cosine));

            if (angle > params.max_segment_angle_rad) {
                breaks.push_back(i);
            }
        }

        // Build split indices
        std::vector<int> split_idxs = {0};
        split_idxs.insert(split_idxs.end(), breaks.begin(), breaks.end());
        split_idxs.push_back(len);

        for (size_t i = 0; i < split_idxs.size() - 1; ++i) {
            int a = split_idxs[i];
            int b = split_idxs[i + 1];
            if (b - a < params.min_segment_length)
                continue;



            segment_points.push_back(std::vector<cv::Point>());
            for (int j = a; j < b; ++j) {
                const cv::Point& point = contour[j];
                if (point.y >= 0 && point.y < contours.rows && point.x >= 0 && point.x < contours.cols) {
                    contours.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(255, 255, 255);
                    segment_points.at(segment_points.size() - 1).push_back(point);
                }
                
               
            }

            ++contour_id;
        }
    }

    // vvv PRINTS FOR DEBUGGING vvv
    // cout << endl << "detected " << contour_id << " contours" << endl;
    // cout << endl << "segments.size(): " << segments.size() << endl;
    // cout << endl << "segments.at(0).size(): " << segments.at(0).size() << endl;
    // for (const auto& pt : segments.at(0)) { 
    //     cout << pt;
    // }

    return contour_id;
}


// TODO move this
void sortSegmentsByLength(std::vector<std::vector<cv::Point>>& segments, std::vector<double>& lengths) {

    // Pair lengths with their corresponding segment
    std::vector<std::pair<double, std::vector<cv::Point>>> paired;

    for (size_t i = 0; i < lengths.size(); ++i) {
    paired.emplace_back(lengths[i], segments[i]);
    }

    // Sort by length (ascending)
    std::sort(paired.begin(), paired.end(),
    [](const auto& a, const auto& b) {
    return a.first > b.first;
    });

    // Unpack back into segments and lengths
    for (size_t i = 0; i < paired.size(); ++i) {
    lengths[i] = paired[i].first;
    segments[i] = std::move(paired[i].second);
    }
}

void Mosaic::rankContours() { 
    if (segment_points.empty()) {
        std::cerr << "rankSegments called but contours image is empty" << std::endl;
        return;
    }

    segment_lengths.clear(); // Before computing

    for (const auto& segment_pixels : segment_points) {
        double length = Geometry::pcaLength(segment_pixels);
        segment_lengths.push_back(length);
    }

    sortSegmentsByLength(segment_points, segment_lengths);


}
































void Mosaic::placeTileAllContours() { 

    // TODO - find the number of segments
    int number_of_segments = static_cast<int>(segment_lengths.size());

    for (int i = 0; i < number_of_segments; i++) {
        placeTileContour(i);
    }

    // cout << "Placed tiles along: " << number_of_segments << " segments" << endl;
    
}




void Mosaic::placeTileContour(int k) { 

    if (mask.empty()) { 
        mask = cv::Mat::zeros(resized.size(), CV_8UC1);
    }

    selectContour(k);


    cv::Point center = getRandomPointOnContour(k);     // TODO we can't just assume the first random spot will be valid for tile placement. (although this works first time)

    double size = params.tile_size;


    stack<cv::Point> s;
    cv::Point current_center;
    s.push(center);
    int squares_placed = 0;
    int frontier = 0;


    while(!s.empty()) { 
        current_center = s.top();
        s.pop();


        
        double theta_deg = findBestTheta(current_center, size);
        if (!isValidTile(current_center, size, theta_deg)) { 
            continue;
        }
        
        placeTile(current_center, size, theta_deg, frontier, std::to_string(squares_placed));

        
        squares_placed++;


   
        double initial_size = size;
        std::vector<cv::Point> allIntersections;


        for (int i = 0; i < params.number_of_rings; ++i) {
            double currentSize = initial_size + i * params.step_size;
        
            std::vector<cv::Point> ringIntersections = findRingIntersections(
                selected_contour, current_center, currentSize, theta_deg
            );
        
            allIntersections.insert(allIntersections.end(), ringIntersections.begin(), ringIntersections.end());
        }


        // filter and sort -- add closest points to top of stack
        // TODO geometry
        allIntersections = filterUniqueIntersections(allIntersections);
        std::sort(allIntersections.begin(), allIntersections.end(),
        [&current_center](const cv::Point& a, const cv::Point& b) {
            return Geometry::euclideanDistance(a, current_center) < Geometry::euclideanDistance(b, current_center);
        });
        std::reverse(allIntersections.begin(), allIntersections.end());

        for (cv::Point p : allIntersections) { 
            s.push(p);
        }

    }


}



void Mosaic::selectContour(int k) {
    


    if (segment_lengths.empty()) {
        std::cerr << "selectSegment called but segment_lengths is empty." << std::endl;
        return;
    }

    if (k < 0 || k >= static_cast<int>(segment_lengths.size())) {
        std::cerr << "selectSegment: k = " << k << " is out of range. Valid range: [0, "
                  << segment_lengths.size() - 1 << "]\n";
        return;
    }

    // Create a blank image
    selected_contour = cv::Mat::zeros(resized.size(), CV_8UC3);


    for (cv::Point point : segment_points.at(k)) {
        // cout << "point.x: " << point.x << " point.y: " << point.y << endl;
        selected_contour.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(255, 255, 255);
    }



}



cv::Point Mosaic::getRandomPointOnContour(int k) {
    // Safety check: make sure k is in bounds
    if (k < 0 || k >= static_cast<int>(segment_lengths.size())) {
        throw std::out_of_range("Segment index k is out of range");
    }

    const std::vector<cv::Point>& points = segment_points.at(k);

    if (points.empty()) {
        throw std::runtime_error("No points in the selected segment");
    }

    return Random::selectFromVector<cv::Point>(points);
}


double Mosaic::findBestTheta(cv::Point center, double size) { 

    double radius = size * 1.0; // HUGE impact on alignment TODO do some geometry

    // Convert to grayscale if needed
    // TODO move into geometry and reuse gray_segment
    cv::Mat gray;
    if (selected_contour.channels() > 1) {
        cv::cvtColor(selected_contour, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = selected_contour;
    }

    // Find non-zero stroke pixels
    std::vector<cv::Point> all_stroke_pixels;
    cv::findNonZero(gray, all_stroke_pixels);

    // Filter to those within circular radius
    std::vector<cv::Point2d> region_pixels;
    for (const auto& pt : all_stroke_pixels) {
        double dx = pt.x - center.x;
        double dy = pt.y - center.y;
        if ((dx * dx + dy * dy) <= radius * radius) {
            region_pixels.emplace_back(pt.x, pt.y);
        }
    }

    // Check if enough points for PCA
    if (region_pixels.size() < 2) {
        return ERROR_CODE_NO_VALID_THETA;
    }

    cv::Vec2d direction = Geometry::pcaDirection(region_pixels);
    double theta_deg = Geometry::vectorToAngleDegrees(direction);

    

    return theta_deg;
}



bool Mosaic::isValidTile(cv::Point center, double size, double theta_deg) {
    // check for validity
    if (theta_deg == ERROR_CODE_NO_VALID_THETA) { 
        return false;
    }
    if (tileOverlapsMask(center, size, theta_deg)) { 
        return false;
    }
    if (!tileInBounds(center, size)) { 
        return false;
    }
    return true;
}



bool Mosaic::tileInBounds(const cv::Point& center, double tileSize) { 
    // Early exit if tile is far outside the image
    // int margin = static_cast<int>(2 * tileSize);
    int margin = 0;

    if (center.x < -margin || center.y < -margin ||
        center.x > mask.cols + margin || center.y > mask.rows + margin) {
        return false;
    }
    return true;
}


bool Mosaic::tileOverlapsMask(const cv::Point& center, double tileSize, double rotationDegrees) {

    // 0. super fast check (5% speedup)
    if (mask.at<uchar>(center) > 0) { 
        return true;
    }

    // 1. Compute tile corners
    float halfSize = static_cast<float>(tileSize / 2.0);
    float theta = static_cast<float>(rotationDegrees * CV_PI / 180.0);

    std::vector<cv::Point2f> localCorners = {
        {-halfSize, -halfSize},
        { halfSize, -halfSize},
        { halfSize,  halfSize},
        {-halfSize,  halfSize}
    };

    cv::Point2f centerF(center);
    std::vector<cv::Point> worldCorners;
    for (const auto& pt : localCorners) {
        float x = pt.x * std::cos(theta) - pt.y * std::sin(theta);
        float y = pt.x * std::sin(theta) + pt.y * std::cos(theta);
        worldCorners.emplace_back(cvRound(centerF.x + x), cvRound(centerF.y + y));
    }

    // 2. FAST CORNER CHECK (40:1 speed up for flood fill)
    for (const auto& pt : worldCorners) {
        if (pt.x >= 0 && pt.x < mask.cols && pt.y >= 0 && pt.y < mask.rows) {
            if (mask.at<uchar>(pt) > 0) {
                return true;
            }
        }
    }

    // 3. Create tile mask and check full overlap
    cv::Mat tileMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contour{worldCorners};
    cv::fillPoly(tileMask, contour, cv::Vec3b(255));

    cv::Mat overlap;
    cv::bitwise_and(mask, tileMask, overlap);

    return cv::countNonZero(overlap) > 0;
}



TileInfo Mosaic::placeTile(cv::Point center, double size, double theta_deg, int frontier, string text) {



    Graphics::drawSquare(mask, center, size, theta_deg, cv::Vec3b(255), size);

    // TODO add tile metadata to the 
    int order = tiles_placed.size();
    TileInfo current_tile = {
        center,
        size, 
        theta_deg,
        order,
        frontier
    };
    tiles_placed.push_back(current_tile);
    tiles_to_render.push_back(current_tile);

    if (tiles_to_render.size() >= params.tiles_per_frame) { 
        renderTiles();
        tiles_to_render.clear();
    }

    return current_tile;

}

void Mosaic::renderTiles() { 

    if (canvas.empty()) { 
        canvas = cv::Mat::zeros(resized.size(), CV_8UC3);
    }


    if (window_name.empty()) { 
        return;
    }

    for (TileInfo tile : tiles_to_render) { 
        cv::Vec3b color = sampleTileColor(tile);
        Graphics::drawSquare(canvas, tile.center, tile.size, tile.theta_deg, color, tile.size);
    }


    cv::imshow(window_name, canvas);
    cv::waitKey(1); // Needed for OpenCV to update GUI
}

// TODO move to geometry
double pointLineSegmentDistance(const cv::Point2f& p, const cv::Point2f& A, const cv::Point2f& B) {
    cv::Point2f AB = B - A;
    cv::Point2f AP = p - A;

    double ab2 = AB.dot(AB);
    if (ab2 == 0.0) return cv::norm(AP); // A == B case

    double t = AP.dot(AB) / ab2;
    t = std::max(0.0, std::min(1.0, t)); // Clamp t to [0,1]

    cv::Point2f projection = A + t * AB;
    return cv::norm(p - projection);
}

std::vector<cv::Point> Mosaic::findRingIntersections(const cv::Mat& segment_image, const cv::Point2f& center, double tileSize, double rotationDegrees) {
    std::vector<cv::Point> intersections;

    // Convert to grayscale if needed
    cv::Mat gray;
    if (segment_image.channels() > 1) {
        cv::cvtColor(segment_image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = segment_image;
    }

    // Threshold to binary mask
    cv::Mat binaryGray;
    cv::threshold(gray, binaryGray, 128, 255, cv::THRESH_BINARY);

    // Compute half-size and rotation radians
    float halfSize = tileSize / 2.0f;
    float theta = rotationDegrees * CV_PI / 180.0f;

    // Define corners in local space
    std::vector<cv::Point2f> localCorners = {
        {-halfSize, -halfSize},
        { halfSize, -halfSize},
        { halfSize,  halfSize},
        {-halfSize,  halfSize}
    };

    // Rotate and translate corners to world space
    std::vector<cv::Point2f> worldCorners(4);
    for (int i = 0; i < 4; ++i) {
        float x = localCorners[i].x;
        float y = localCorners[i].y;
        float xr = x * std::cos(theta) - y * std::sin(theta);
        float yr = x * std::sin(theta) + y * std::cos(theta);
        worldCorners[i] = center + cv::Point2f(xr, yr);
    }

    // Create tile mask from polygon of rotated corners
    cv::Mat tileMask = cv::Mat::zeros(gray.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contour(1);
    for (const auto& pt : worldCorners)
        contour[0].push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));

    cv::fillPoly(tileMask, contour, cv::Vec3b(255));

    // Bitwise AND between binary segment mask and tile mask
    cv::Mat overlap;
    cv::bitwise_and(binaryGray, tileMask, overlap);

    // Find all non-zero points inside tile
    cv::findNonZero(overlap, intersections);

    // Filter intersections to keep only points near edges
    const double borderThreshold = 3.0; // pixels
    std::vector<cv::Point> borderIntersections;

    for (const auto& pt : intersections) {
        cv::Point2f p(pt.x, pt.y);

        bool nearEdge = false;
        for (int i = 0; i < 4; ++i) {
            const cv::Point2f& A = worldCorners[i];
            const cv::Point2f& B = worldCorners[(i + 1) % 4];

            double dist = pointLineSegmentDistance(p, A, B);
            if (dist <= borderThreshold) {
                nearEdge = true;
                break;
            }
        }

        if (nearEdge) {
            borderIntersections.push_back(pt);
        }
    }

    return borderIntersections;
}



std::vector<cv::Point> Mosaic::filterUniqueIntersections(const std::vector<cv::Point>& inputPoints) {
    

    const double min_dist = params.min_intersection_distance; 

    return Geometry::filterUniquePoints(inputPoints, min_dist);
}



























void Mosaic::floodFill() {

    double distance_from_center = params.distance_from_center;
    const int max_frontiers = params.max_frontiers;
    int frontier = 1;
    std::vector<TileInfo> frontier_tiles(tiles_placed);

    while (!frontier_tiles.empty() && frontier <= max_frontiers) { 
        
        // Collect all flood fill points for this frontier
        std::vector<cv::Point> all_flood_points;
        for (const TileInfo& tile : frontier_tiles) {
            int num_points = params.flood_fill_neighbor_points;
            int max_step = getJitter(frontier);
            std::vector<cv::Point> points = nextFrontierFromTile(tile.center, tile.theta_deg, distance_from_center, num_points);
            std::vector<cv::Point> jittered_points = Random::jitterPoints(points, max_step, mask.size());
            all_flood_points.insert(all_flood_points.end(), jittered_points.begin(), jittered_points.end());
        }

        // TODO sort somehow

        std::vector<TileInfo> next_frontier_tiles;
        // Now place tiles at unique positions
        for (const cv::Point& pt : all_flood_points) {
            double theta_deg = findBestThetaTangentField(pt);
            if (isValidTile(pt, params.tile_size, theta_deg)) { 
                TileInfo current_tile = placeTile(pt, params.tile_size, theta_deg, frontier + 1);
                next_frontier_tiles.push_back(current_tile);
            }
            
        }

        frontier_tiles = next_frontier_tiles;
        frontier++;
    }
}



std::vector<cv::Point> Mosaic::nextFrontierFromTile(cv::Point center, double theta_deg, double distance_from_center, int num_points) {
    std::vector<cv::Point> flood_points;

    if (num_points <= 0) return flood_points;

    // Convert center to float
    cv::Point2f center_f(center.x, center.y);

    // Convert angle to radians
    double theta_rad = theta_deg * CV_PI / 180.0;

    // Rotated basis vectors
    cv::Point2f dx(std::cos(theta_rad), std::sin(theta_rad));       // along width
    cv::Point2f dy(-std::sin(theta_rad), std::cos(theta_rad));      // along height

    // Define half-size
    double h = distance_from_center;

    // Corners of the square (in local unrotated coordinates)
    std::vector<cv::Point2f> square = {
        cv::Point2f(-h, -h),  // top-left
        cv::Point2f(h, -h),   // top-right
        cv::Point2f(h, h),    // bottom-right
        cv::Point2f(-h, h)    // bottom-left
    };

    // Total perimeter of square
    double perimeter = 8 * h;

    for (int i = 0; i < num_points; ++i) {
        double t = (i / (double)num_points) * perimeter;

        cv::Point2f local;

        if (t < 2 * h) {
            // Top edge
            local = square[0] + cv::Point2f(t, 0);
        } else if (t < 4 * h) {
            // Right edge
            local = square[1] + cv::Point2f(0, t - 2 * h);
        } else if (t < 6 * h) {
            // Bottom edge
            local = square[2] + cv::Point2f(-(t - 4 * h), 0);
        } else {
            // Left edge
            local = square[3] + cv::Point2f(0, -(t - 6 * h));
        }

        // Rotate the local point using the dx/dy basis
        cv::Point2f rotated = center_f + dx * local.x + dy * local.y;

        // Convert to integer and store
        flood_points.push_back(cv::Point(cvRound(rotated.x), cvRound(rotated.y)));
    }

    return flood_points;
}


double Mosaic::findBestThetaTangentField(cv::Point center) { 
    auto [tangent, dist] = getTangentAtPoint(center);
    double theta_rad = std::atan2(tangent[1], tangent[0]);
    double theta_deg = theta_rad * 180.0 / CV_PI;


    return theta_deg;
};



std::tuple<cv::Vec2d, float> Mosaic::getTangentAtPoint(const cv::Point& pt) {

    if (distance_map.empty() || gradX.empty() || gradY.empty()) { 
        computeDistanceField();
    }

    int x = pt.x;
    int y = pt.y;

    if (x < 0 || x >= distance_map.cols || y < 0 || y >= distance_map.rows) {
    return {cv::Vec2d(0.0f, 0.0f), 0.0f};
    }

    float dx = gradX.at<float>(y, x);
    float dy = gradY.at<float>(y, x);

    // Rotate 90° counter-clockwise to get tangent
    float tx = -dy;
    float ty = dx;

    float magnitude = std::sqrt(tx * tx + ty * ty);
    cv::Vec2d tangent(0.0f, 0.0f);
    if (magnitude > 1e-5f) {
    tangent = cv::Vec2d(tx / magnitude, ty / magnitude);
    }

    float dist = distance_map.at<float>(y, x);
    return {tangent, dist};
}



void Mosaic::computeDistanceField() { 

    distance_map = cv::Mat::zeros(contours.size(), CV_8UC3);
    gradX = cv::Mat::zeros(contours.size(), CV_8UC3);
    gradY = cv::Mat::zeros(contours.size(), CV_8UC3);


    // Step 1: Convert to grayscale and binary edge map
    cv::Mat gray, binary;
    cv::cvtColor(contours, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 1, 255, cv::THRESH_BINARY);

    // Step 2: Invert binary
    cv::Mat inverted = 255 - binary;

    // Step 3: Compute distance transform
    cv::distanceTransform(inverted, distance_map, cv::DIST_L2, 3);

    // Step 4: Compute gradients
    cv::Sobel(distance_map, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(distance_map, gradY, CV_32F, 0, 1, 3);


}


















void Mosaic::gapFill() { 
    if (distance_map.empty() || gradX.empty() || gradY.empty()) { 
        computeDistanceField();
    }

    double tile_size = params.tile_size;
    int max_tiles_to_place = params.random_background_points;

    std::vector<cv::Point> points;
    for (int y = 0; y < distance_map.rows; ++y) {
        for (int x = 0; x < distance_map.cols; ++x) {
            float dist = distance_map.at<float>(y, x);
            if (dist >= tile_size * 0.5) { // threshold: skip very narrow gaps
                points.push_back(cv::Point(x, y));
            }
        }
    }
    

    Random::shuffleVector(points);



    int num_tiles_placed = 0;

   for (const cv::Point& point : points) {


        // Sample guidance field
        auto [vec, contour_dist] = getTangentAtPoint(point);
        double theta_deg = Geometry::vectorToAngleDegrees(vec);

        if (isValidTile(point, tile_size, theta_deg)) {

            int frontier = -1;
            placeTile(point, tile_size, theta_deg, frontier); 
            ++num_tiles_placed;
        }
    }

    // std::cout << "Filled " << num_tiles_placed << " gaps using distance field.\n";
}




















void Mosaic::reconstructImage() { 

    // reset canvas
    canvas = cv::Mat::zeros(edges.size(), CV_8UC3);
    cv::Vec3b color;

    for (TileInfo tile : tiles_placed) { 
        color = sampleTileColor(tile);
        Graphics::drawSquare(canvas, tile.center, tile.size * 1.0, tile.theta_deg, color, tile.size);
    }


}




cv::Vec3b Mosaic::sampleTileColor(const TileInfo& tile) {
    float halfSize = static_cast<float>(tile.size / 2.0);
    float theta = static_cast<float>(tile.theta_deg * CV_PI / 180.0f);

    std::vector<cv::Point2f> localCorners = {
        {-halfSize, -halfSize},
        { halfSize, -halfSize},
        { halfSize,  halfSize},
        {-halfSize,  halfSize}
    };

    std::vector<cv::Point> worldCorners;
    for (const auto& pt : localCorners) {
        float x = pt.x * std::cos(theta) - pt.y * std::sin(theta);
        float y = pt.x * std::sin(theta) + pt.y * std::cos(theta);
        worldCorners.emplace_back(cvRound(tile.center.x + x), cvRound(tile.center.y + y));
    }

    // Create mask for the rotated tile
    cv::Mat mask = cv::Mat::zeros(resized.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contour{worldCorners};
    cv::fillPoly(mask, contour, cv::Vec3b(255));

    // Return average color from resized image under tile
    cv::Scalar color = cv::mean(resized, mask);


    cv::Vec3b vecColor(
        static_cast<uchar>(color[0]),
        static_cast<uchar>(color[1]),
        static_cast<uchar>(color[2])
    );


    return vecColor;
}





}



