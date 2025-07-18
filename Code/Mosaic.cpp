#include "Mosaic.hpp"
#include "graphics.hpp"
#include "optimize.hpp"

#include "gif.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <filesystem>
#include <stack>
#include <sstream>
#include <fstream>

using namespace std;
namespace fs = std::__fs::filesystem;

namespace mosaic_gen {

// param constructor
Mosaic::Mosaic(const HyperParameters& hp) { 
    params = hp;
}



void Mosaic::loadImage() { 
    string image_path = params.image_path;
    original = cv::imread(image_path);

    if (original.empty()) { 
        cerr << "Error: Could not load image from path: " << image_path << endl;
        return;
    }

    file_path = image_path;
    image_name = fs::path(image_path).stem().string();

}

// set hyperparameters



void Mosaic::resizeOriginal() { 
    if (original.empty()) { 
        cerr << "Resized called but no original image found" << endl;
        return;
    }

    cv::resize(original, resized, cv::Size(), params.resize_factor, params.resize_factor, cv::INTER_LINEAR);


    // construct image mats


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


int Mosaic::detectContours() { 
    if (edges.empty()) {
        cerr << "DetectContours called but no edges" << endl;
        return -1;
    }

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Create an output color image
    segmented = cv::Mat::zeros(edges.size(), CV_8UC3);
    int contour_id = 0;
    std::vector<cv::Vec3b> colors_used;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> color_dist(64, 255);

    for (const auto& contour : contours) {
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

            // Generate a new color not used yet
            cv::Vec3b color;
            do {
                color = cv::Vec3b(color_dist(rng), color_dist(rng), color_dist(rng));
            } while (std::find(colors_used.begin(), colors_used.end(), color) != colors_used.end());

            colors_used.push_back(color);

            for (int j = a; j < b; ++j) {
                const auto& pt = contour[j];
                if (pt.y >= 0 && pt.y < segmented.rows && pt.x >= 0 && pt.x < segmented.cols) {
                    segmented.at<cv::Vec3b>(pt.y, pt.x) = color;
                }
            }

            ++contour_id;
        }
    }

    return contour_id;
}

void Mosaic::rankSegments() { 
    if (segmented.empty()) {
        std::cerr << "rankSegments called but segmented image is empty" << std::endl;
        return;
    }

    segment_pixels.clear();
    segment_lengths.clear();

    // Collect pixels for each color (excluding black)
    for (int y = 0; y < segmented.rows; ++y) {
        for (int x = 0; x < segmented.cols; ++x) {
            cv::Vec3b color = segmented.at<cv::Vec3b>(y, x);
            if (color != cv::Vec3b(0, 0, 0)) {
                segment_pixels[color].emplace_back(x, y);
            }
        }
    }

    // Helper lambda for PCA length
    auto pca_length = [](const std::vector<cv::Point>& points) -> double {
        if (points.size() < 2)
            return 0.0;

        cv::Mat data(points.size(), 2, CV_64F);
        for (size_t i = 0; i < points.size(); ++i) {
            data.at<double>(i, 0) = points[i].x;
            data.at<double>(i, 1) = points[i].y;
        }

        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
        cv::Mat projected;
        pca.project(data, projected);

        double minVal, maxVal;
        cv::minMaxLoc(projected.col(0), &minVal, &maxVal);

        return maxVal - minVal;
    };

    // Compute PCA length per color segment
    for (const auto& [color, pixels] : segment_pixels) {
        double length = pca_length(pixels);
        segment_lengths.emplace_back(color, length);
    }

    // Sort descending by length
    std::sort(segment_lengths.begin(), segment_lengths.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
}


void Mosaic::selectSegment(int k) { 
    if (segment_lengths.empty()) {
        std::cerr << "selectSegment called but segment_lengths is empty." << std::endl;
        return;
    }

    if (k < 0 || k >= static_cast<int>(segment_lengths.size())) {
        std::cerr << "selectSegment: k = " << k << " is out of range. Valid range: [0, "
                  << segment_lengths.size() - 1 << "]\n";
        return;
    }

    const cv::Vec3b& selected_color = segment_lengths[k].first;

    auto it = segment_pixels.find(selected_color);
    if (it == segment_pixels.end()) {
        std::cerr << "selectSegment: Selected color not found in segment_pixels.\n";
        return;
    }

    // Create a blank image
    selected_segment = cv::Mat::zeros(segmented.size(), CV_8UC3);

    // Draw only the selected segment
    for (const auto& pt : it->second) {
        if (pt.y >= 0 && pt.y < selected_segment.rows && pt.x >= 0 && pt.x < selected_segment.cols) {
            selected_segment.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(255, 255, 255);
        }
    }
}


cv::Point Mosaic::getRandomPointOnSegment(int k) {
    // Safety check: make sure k is in bounds
    if (k < 0 || k >= static_cast<int>(segment_lengths.size())) {
        throw std::out_of_range("Segment index k is out of range");
    }

    // Get the color for segment k (assuming segment_lengths[k].first is the color)
    const cv::Vec3b& color = segment_lengths[k].first;

    // Find the vector of points corresponding to this color
    auto it = segment_pixels.find(color);
    if (it == segment_pixels.end()) {
        throw std::runtime_error("Color not found in segment_pixels");
    }

    const std::vector<cv::Point>& points = it->second;

    if (points.empty()) {
        throw std::runtime_error("No points in the selected segment");
    }

    // Random engine and distribution
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, static_cast<int>(points.size()) - 1);

    // Pick a random index and return the point
    return points[dist(rng)];
}




void Mosaic::drawSquareRandomPoint(int k) { 

    cv::Point center = getRandomPointOnSegment(k);
    cv::Scalar color = cv::Scalar(255, 0, 255);

    canvas = selected_segment.clone();
    Graphics::drawSquare(canvas, center, 40.0, 12.0, color, 2.0);


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

    // 2. FAST CORNER CHECK (early exit if any corner already marked in mask)
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
    cv::fillPoly(tileMask, contour, cv::Scalar(255));

    cv::Mat overlap;
    cv::bitwise_and(mask, tileMask, overlap);

    return cv::countNonZero(overlap) > 0;
}








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

std::vector<cv::Point> Mosaic::findTileEdgeIntersections(const cv::Mat& segment_image, const cv::Point2f& center, double tileSize, double rotationDegrees) {
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

    cv::fillPoly(tileMask, contour, cv::Scalar(255));

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





double euclideanDistance(const cv::Point& a, const cv::Point& b) {
    double dx = static_cast<double>(a.x - b.x);
    double dy = static_cast<double>(a.y - b.y);
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<cv::Point> Mosaic::filterUniqueIntersections(const std::vector<cv::Point>& inputPoints) {
    std::vector<cv::Point> uniquePoints;

    const double MIN_INTERSECTION_DIST = 10.0;  // adjust as needed

    for (const auto& pt : inputPoints) {
        bool isFarEnough = true;
        for (const auto& kept : uniquePoints) {
            if (euclideanDistance(pt, kept) < MIN_INTERSECTION_DIST) {
                isFarEnough = false;
                break;
            }
        }
        if (isFarEnough) {
            uniquePoints.push_back(pt);
        }
    }

    return uniquePoints;
}




double Mosaic::findBestTheta(cv::Point center, double size) { 

    double radius = size * 1.0; // HUGE impact on alignment TODO do some geometry

    // Convert to grayscale if needed
    cv::Mat gray;
    if (selected_segment.channels() > 1) {
        cv::cvtColor(selected_segment, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = selected_segment;
    }

    // Find non-zero stroke pixels
    std::vector<cv::Point> all_stroke_pixels;
    cv::findNonZero(gray, all_stroke_pixels);

    // Filter to those within circular radius
    std::vector<cv::Point2f> region_pixels;
    for (const auto& pt : all_stroke_pixels) {
        double dx = pt.x - center.x;
        double dy = pt.y - center.y;
        if ((dx * dx + dy * dy) <= radius * radius) {
            region_pixels.emplace_back(pt.x, pt.y);
        }
    }

    // Check if enough points for PCA
    if (region_pixels.size() < 2) {
        std::cerr << "Not enough stroke pixels for PCA near point: " << center << std::endl;
        return ERROR_CODE_NO_VALID_THETA;
    }

    // Build matrix for PCA
    cv::Mat data(region_pixels.size(), 2, CV_64F);
    for (size_t i = 0; i < region_pixels.size(); ++i) {
        data.at<double>(i, 0) = region_pixels[i].x;
        data.at<double>(i, 1) = region_pixels[i].y;
    }

    // Run PCA to get dominant direction
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
    cv::Vec2d direction = pca.eigenvectors.row(0);

    // Convert to angle in degrees
    double theta_rad = std::atan2(direction[1], direction[0]);
    double theta_deg = theta_rad * 180.0 / CV_PI;

    // cout << "Best theta: " << theta_deg << endl;


    // check for validity
    if (tileOverlapsMask(center, size, theta_deg)) { 
        return ERROR_CODE_NO_VALID_THETA;
    }
    if (!tileInBounds(center, size)) { 
        return ERROR_CODE_NO_VALID_THETA;
    }

    return theta_deg;
}



// Returns theta_deg if placed, -420 if not placed
double Mosaic::placeTile(cv::Point center, double size, string text) {

    // findBestTheta handles overlap check & pca error
    double theta_deg = findBestTheta(center, size);

    // pass error so dfs stops
    if (theta_deg == ERROR_CODE_NO_VALID_THETA) { 
        return ERROR_CODE_NO_VALID_THETA;
    }

    // Draw aligned square
    cv::Scalar color = cv::Scalar(255, 255, 0);
    Graphics::drawSquareText(canvas, center, size, theta_deg, color, 2.0, text);
    Graphics::drawSquare(mask, center, size, theta_deg, color, 2.0);

    // TODO add tile metadata to the 
    int order = tiles_placed.size();
    int frontier = 0; // contour trace is frontier 0
    TileInfo current_tile = {
        center,
        size, 
        theta_deg,
        order,
    };
    tiles_placed.push_back(current_tile);

    return theta_deg;






}



void Mosaic::placeTileSegment(int k) { 

    if (mask.empty()) { 
        mask = cv::Mat::zeros(resized.size(), CV_8UC1);
    }

    selectSegment(k);
    canvas = selected_segment.clone();

    cv::Point center = getRandomPointOnSegment(k);     // TODO we can't just assume the first random spot will be valid for tile placement. (although this works first time)

    double size = params.tile_size;
    cv::Scalar color(255, 255, 0);





    

    stack<cv::Point> s;
    cv::Point current_center;
    s.push(center);
    int squares_placed = 0;


    while(!s.empty()) { 
        current_center = s.top();
        s.pop();

        // cout << "current center: " << current_center << endl;

        double theta_deg = placeTile(current_center, size, std::to_string(squares_placed));
        // no valid placement
        if (theta_deg == ERROR_CODE_NO_VALID_THETA) { 
            // cout << "no valid theta" << endl;
            continue;
        }
        squares_placed++;


   
        double initial_size = size;

        std::vector<cv::Point> allIntersections;

        for (int i = 0; i < params.number_of_rings; ++i) {
            double currentSize = initial_size + i * params.step_size;


            std::vector<cv::Point> ringIntersections = findTileEdgeIntersections(
                selected_segment, current_center, currentSize, theta_deg
            );

            allIntersections.insert(allIntersections.end(), ringIntersections.begin(), ringIntersections.end());
        }

        allIntersections = filterUniqueIntersections(allIntersections);
        // cout << "number of intersections kept after filter: " << allIntersections.size() << endl;




        // sort and add closest points to top of stack
        std::sort(allIntersections.begin(), allIntersections.end(),
        [&current_center](const cv::Point& a, const cv::Point& b) {
            return euclideanDistance(a, current_center) < euclideanDistance(b, current_center);
        });
        std::reverse(allIntersections.begin(), allIntersections.end());

        for (cv::Point p : allIntersections) { 
            s.push(p);
        }

    }

    









}




// TODO finish this function
void Mosaic::placeTileAllSegments() { 

    // TODO - find the number of segments
    int number_of_segments = static_cast<int>(segment_lengths.size());

    for (int i = 0; i < number_of_segments; i++) {
        placeTileSegment(i);
    }

    cout << "Placed tiles along: " << number_of_segments << " segments" << endl;
    
}








cv::Scalar Mosaic::sampleTileColor(const TileInfo& tile) {
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
    cv::fillPoly(mask, contour, cv::Scalar(255));

    // Return average color from resized image under tile
    return cv::mean(resized, mask);
}


void Mosaic::reconstructPlacedTiles() { 

    // reset canvas
    canvas = cv::Mat::zeros(edges.size(), CV_8UC3);
    cv::Scalar color;

    for (TileInfo tile : tiles_placed) { 
        color = sampleTileColor(tile);
        Graphics::drawSquare(canvas, tile.center, tile.size, tile.theta_deg, color, tile.size);
    }


}



std::vector<cv::Point> Mosaic::samplePointsGrid(const cv::Mat& image, int grid_size) { 
    std::vector<cv::Point> grid_points;

    if (image.empty() || grid_size <= 0) {
        return grid_points;
    }

    for (int y = 0; y < image.rows; y += grid_size) {
        for (int x = 0; x < image.cols; x += grid_size) {
            grid_points.emplace_back(x, y);
        }
    }

    return grid_points;
}

std::vector<cv::Point> Mosaic::samplePointsRandom(const cv::Mat& image, int num_points) {
    std::vector<cv::Point> random_points;

    if (image.empty() || num_points <= 0) {
        return random_points;
    }

    int width = image.cols;
    int height = image.rows;

    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_x(0, width - 1);
    std::uniform_int_distribution<int> dist_y(0, height - 1);

    for (int i = 0; i < num_points; ++i) {
        int x = dist_x(rng);
        int y = dist_y(rng);
        random_points.emplace_back(x, y);
    }

    return random_points;
}

std::vector<cv::Point> Mosaic::samplePointsRandomGrid(const cv::Mat& image, int grid_size, int max_step) {
    std::vector<cv::Point> jittered_points;

    if (image.empty() || grid_size <= 0 || max_step < 0) {
        return jittered_points;
    }

    int width = image.cols;
    int height = image.rows;

    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> offset_dist(-max_step, max_step);

    for (int y = 0; y < height; y += grid_size) {
        for (int x = 0; x < width; x += grid_size) {
            int jitter_x = x + offset_dist(rng);
            int jitter_y = y + offset_dist(rng);

            // Clamp to image bounds
            jitter_x = std::clamp(jitter_x, 0, width - 1);
            jitter_y = std::clamp(jitter_y, 0, height - 1);

            jittered_points.emplace_back(jitter_x, jitter_y);
        }
    }

    return jittered_points;
}



















void Mosaic::placeTileBackground(cv::Point center, double size, double theta_deg, int frontier) {




    // check for validity
    if (tileOverlapsMask(center, size, theta_deg)) { 
        return;
    }

    if (!tileInBounds(center, size)) {
        return;
    }

    // Draw aligned square
    // cv::Scalar color = cv::Scalar(150, 150, 0);
    cv::Scalar color = cv::Scalar(255, 255, 255);
    Graphics::drawSquareText(canvas, center, size, theta_deg, color, 2.0, std::to_string(frontier));
    Graphics::drawSquare(mask, center, size, theta_deg, color, 2.0);

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


}

void Mosaic::placeTileAllBackground() {
    if (tiles_placed.empty()) {
        std::cerr << "No tiles placed to map samples to." << std::endl;
        return;
    }

    
    int num_points = 50000;
    std::vector<cv::Point> samples = samplePointsRandom(canvas, num_points);




    for (const auto& sample : samples) {

        auto [tangent, _] = sampleTangentPoint(sample);
        double theta_rad = std::atan2(tangent[1], tangent[0]);
        double theta_deg = theta_rad * 180.0 / CV_PI;
        

        placeTileBackground(sample, params.tile_size, theta_deg);
    }
}















void Mosaic::computeDistanceField() { 

    distance = cv::Mat::zeros(segmented.size(), CV_8UC3);
    gradX = cv::Mat::zeros(segmented.size(), CV_8UC3);
    gradY = cv::Mat::zeros(segmented.size(), CV_8UC3);


    // Step 1: Convert to grayscale and binary edge map
    cv::Mat gray, binary;
    cv::cvtColor(segmented, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 1, 255, cv::THRESH_BINARY);

    // Step 2: Invert binary
    cv::Mat inverted = 255 - binary;

    // Step 3: Compute distance transform
    cv::distanceTransform(inverted, distance, cv::DIST_L2, 3);

    // Step 4: Compute gradients
    cv::Sobel(distance, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(distance, gradY, CV_32F, 0, 1, 3);

    

}




std::tuple<cv::Vec2f, float> Mosaic::sampleTangentPoint(const cv::Point& pt) {

    if (distance.empty() || gradX.empty() || gradY.empty()) { 
        computeDistanceField();
    }

    int x = pt.x;
    int y = pt.y;

    if (x < 0 || x >= distance.cols || y < 0 || y >= distance.rows) {
    return {cv::Vec2f(0.0f, 0.0f), 0.0f};
    }

    float dx = gradX.at<float>(y, x);
    float dy = gradY.at<float>(y, x);

    // Rotate 90° counter-clockwise to get tangent
    float tx = -dy;
    float ty = dx;

    float magnitude = std::sqrt(tx * tx + ty * ty);
    cv::Vec2f tangent(0.0f, 0.0f);
    if (magnitude > 1e-5f) {
    tangent = cv::Vec2f(tx / magnitude, ty / magnitude);
    }

    float dist = distance.at<float>(y, x);
    return {tangent, dist};
}

double Mosaic::findBestThetaTangentField(cv::Point center) { 
    auto [tangent, dist] = sampleTangentPoint(center);
    double theta_rad = std::atan2(tangent[1], tangent[0]);
    double theta_deg = theta_rad * 180.0 / CV_PI;
    return theta_deg;
};


std::vector<std::tuple<cv::Point, cv::Vec2f, float>> Mosaic::sampleTangentField() {

    


    // int grid_size = 15;
    // std::vector<cv::Point> samplePoints = samplePointsGrid(segmented, grid_size);

    // int num_points = 1000;
    // std::vector<cv::Point> samplePoints = samplePointsRandom(segmented, num_points);

    int grid_size = 25;
    int max_step = 4;
    std::vector<cv::Point> samplePoints = samplePointsRandomGrid(segmented, grid_size, max_step);


    std::vector<std::tuple<cv::Point, cv::Vec2f, float>> results;

    if (segmented.empty() || samplePoints.empty()) {
        return results;
    }

    
    // Step 5: Evaluate tangent and distance at each point
    for (const cv::Point& pt : samplePoints) {
        auto [tangent, dist] = sampleTangentPoint(pt);
        results.emplace_back(pt, tangent, dist);
    }

    // Visualization
    vector_field = cv::Mat::zeros(segmented.size(), CV_8UC3);
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

    


    return results;
}








std::vector<cv::Point> Mosaic::getFloodFillPoints(cv::Point center, double theta_deg, double distance_from_center) {

    cv::Point2f center_f(center.x, center.y);

    std::vector<cv::Point> flood_points;

    // Convert degrees to radians
    double theta_rad = theta_deg * CV_PI / 180.0;

    // Unit vectors in square's rotated X and Y directions
    cv::Point2f dx(std::cos(theta_rad), std::sin(theta_rad));       // direction along tile width
    cv::Point2f dy(-std::sin(theta_rad), std::cos(theta_rad));      // direction along tile height

    // Offset distance from center to each direction
    double offset = distance_from_center;

    // Compute four flood fill target points
    cv::Point2f right  = center_f + dx * offset;
    cv::Point2f left   = center_f - dx * offset;
    cv::Point2f down   = center_f + dy * offset;
    cv::Point2f up     = center_f - dy * offset;

    // Round to integer points for display
    flood_points.push_back(cv::Point(cvRound(right.x), cvRound(right.y)));
    flood_points.push_back(cv::Point(cvRound(left.x), cvRound(left.y)));
    flood_points.push_back(cv::Point(cvRound(down.x), cvRound(down.y)));
    flood_points.push_back(cv::Point(cvRound(up.x), cvRound(up.y)));

    return flood_points;
}




std::vector<cv::Point> Mosaic::getFloodFillPoints2(cv::Point center, double theta_deg, double distance_from_center, int num_points) {
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



void Mosaic::showFloodFillPoints() {
    flood_fill_canvas = cv::Mat::zeros(resized.size(), CV_8UC3);
    double distance_from_center = params.tile_size * 1.5;

    const int max_frontiers = params.max_frontiers;
    for (int frontier = 0; frontier < max_frontiers; frontier++) { 
        
        std::vector<TileInfo> frontier_tiles;
        for (const TileInfo& tile : tiles_placed) {
            if (tile.frontier == frontier) {
                frontier_tiles.push_back(tile);
            }
        }

        cout << frontier_tiles.size() << " tiles on frontier: " << frontier << endl;
        if (frontier_tiles.size() == 0) { 
            return;
        }

        // Collect all flood fill points for this frontier
        std::vector<cv::Point> all_flood_points;
        for (const TileInfo& tile : frontier_tiles) {
            int num_points = 16;
            std::vector<cv::Point> points = getFloodFillPoints2(tile.center, tile.theta_deg, distance_from_center, num_points);
            all_flood_points.insert(all_flood_points.end(), points.begin(), points.end());
        }


        // Now place tiles at unique positions
        for (const cv::Point& pt : all_flood_points) {
            double theta_deg = findBestThetaTangentField(pt);
            if (!tileOverlapsMask(pt, params.tile_size, theta_deg)) { 
                placeTileBackground(pt, params.tile_size, theta_deg, frontier + 1);
            }
            
        }
    }
}




















/*
PRINT FUNCTIONS >>
*/

// Helper to print Vec3b as (B,G,R)
std::string Mosaic::vec3bToString(const cv::Vec3b& color) {
    return "(" + std::to_string(color[0]) + ", " + 
                 std::to_string(color[1]) + ", " + 
                 std::to_string(color[2]) + ")";
}

// Helper to print Point as (x,y)
std::string Mosaic::pointToString(const cv::Point& pt) {
    return "(" + std::to_string(pt.x) + ", " + std::to_string(pt.y) + ")";
}

// Print color_to_pixels unordered_map
void Mosaic::printColorToPixels() {
    std::cout << "Color to Pixels Map:\n";
    for (const auto& [color, points] : segment_pixels) {
        std::cout << "  Color " << vec3bToString(color) << " -> [";
        for (size_t i = 0; i < std::min(points.size(), size_t(5)); ++i) {
            std::cout << pointToString(points[i]);
            if (i != std::min(points.size(), size_t(5)) - 1) std::cout << ", ";
        }
        if (points.size() > 5) std::cout << "...";
        std::cout << "] (" << points.size() << " points)\n";
    }
}


// Print color_lengths vector
void Mosaic::printColorLengths() {
    std::cout << "Color Lengths:\n";
    for (const auto& [color, length] : segment_lengths) {
        std::cout << "  Color " << vec3bToString(color) << " -> Length: " << length << "\n";
    }
}


void Mosaic::printColorToPixelsK(int k) {
    int count = 0;
    std::cout << "Color to Pixels Map:\n";
    for (const auto& [color, points] : segment_pixels) {
        if (count >= k) { 
            break;
        }
        std::cout << "  Color " << vec3bToString(color) << " -> [";
        for (size_t i = 0; i < std::min(points.size(), size_t(5)); ++i) {
            std::cout << pointToString(points[i]);
            if (i != std::min(points.size(), size_t(5)) - 1) std::cout << ", ";
        }
        if (points.size() > 5) std::cout << "...";
        std::cout << "] (" << points.size() << " points)\n";
        count++;
    }
}


void Mosaic::printColorLengthsK(int k) {
    int count = 0;
    std::cout << "Color Lengths:\n";
    for (const auto& [color, length] : segment_lengths) {
        if (count >= k) { 
            break;
        }
        std::cout << "  Color " << vec3bToString(color) << " -> Length: " << length << "\n";
        count++;
    }
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
            cv::Scalar color = sampleTileColor(tile);
            Graphics::drawSquare(gifCanvas, tile.center, tile.size, tile.theta_deg, color, tile.size);
        }

        // Convert to RGBA
        cv::Mat rgba;
        cv::cvtColor(gifCanvas, rgba, cv::COLOR_BGR2RGBA);

        GifWriteFrame(&writer, rgba.data, width, height, 10);
    }

    GifEnd(&writer);
    std::cout << "Saved animated GIF to: " << gifFilename << std::endl;
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
        std::cout << "CSV data successfully written to " << fileName << std::endl;
    } else {
        std::cerr << "Error: Unable to open file '" << fileName << "' for writing." << std::endl;
    }



}











}



