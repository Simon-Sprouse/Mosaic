#include "Mosaic.hpp"

#include "../../Utils/Graphics/graphics.hpp"
#include "../../Utils/Geometry/geometry.hpp"
#include "../../Utils/Random/random.hpp"
#include "../../Utils/ImageProcess/imageProcess.hpp"

// #include "gif.h" TODO add this later

// #include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stack>
#include <queue>

using namespace std;


namespace mosaic_gen {





void Mosaic::runAll() {
  
    contourPipeline();
    placeTilesAllStrokes();
    floodFill();
    gapFill();
    reconstructImage();
}

Image Mosaic::getCanvas() { 
    return canvas.clone();
}

Image* Mosaic::getCanvasPtr() { 
    return &canvas;
}

Image* Mosaic::getDebugCanvasPtr() { 
    return &debugCanvas;
}

Image* Mosaic::getStrokesImagePtr() { 
    return &strokes_image;
}

Image* Mosaic::getOriginalImagePtr() { 
    return &resized;
}

Image Mosaic::getContourImage() {
    Image output(resized.size());
    Graphics::drawStrokesRandomColor(output, strokes);
    return output;
}

uint8_t* Mosaic::getRawData() { 
    return canvas.rawData();
}

bool Mosaic::empty() { 
    return original.empty();
}

Size Mosaic::size() { 
    if (!resized.empty()) return resized.size();
    return Size();
}



void Mosaic::loadImageFromBuffer(const uint8_t *data, size_t size) { 
    original = image::fromEncodedBuffer(data, size);
    image::process::resize(original, resized, params.resize_factor);

    // cout << "inside Mosaic: resized.size(): " << resized.size() << endl;
}

void Mosaic::loadImageFromVector(const std::vector<uint8_t>& buffer){
    original = image::fromEncodedBuffer(buffer.data(), buffer.size());
    image::process::resize(original, resized, params.resize_factor);

}

void Mosaic::loadExistingImage(const Image& img) { 
    original = img.clone();
    image::process::resize(original, resized, params.resize_factor);
}







void Mosaic::contourPipeline() {

    // helper side effects
    Image gray;
    Image blurred;

    std::vector<std::vector<Point>> contours;

    // original = io::loadImage(params.image_path); // done from standalone function now
    // image::process::resize(original, resized, params.resize_factor);

    Size target_size = original.size() * params.resize_factor;
    if (resized.size() != target_size) { 
        image::process::resize(original, resized, target_size);
        // cout << "resized image to: " << resized.size() << endl;
    }
    image::process::grayscale(resized, gray);
    image::process::gaussianBlur(gray, blurred, params.blur_kernel_size, params.blur_sigma);


    // double canny_resize_factor = 0.25;
    double reverse_canny_resize_factor = 1 / params.canny_resize_factor;
    Image resized_for_canny;
    Image canny_downsampled;
    image::process::resize(blurred, resized_for_canny, params.canny_resize_factor);
    image::process::cannyFilter(resized_for_canny, canny_downsampled, params.canny_threshold_1, params.canny_threshold_2);
    image::process::resize(canny_downsampled, canny, reverse_canny_resize_factor);

    image::process::findContours(canny, contours);
    image::process::divideIntoStrokes(contours, strokes, canny.size(), params.segment_angle_window, params.max_segment_angle_rad, params.min_segment_length);
    Geometry::sortStrokesPCALength(strokes);

    strokes_image = Image(resized.size());
    Graphics::drawStrokesRandomColor(strokes_image, strokes);

    // could be separate function, this can run concurrent with stroke covering
    computeDistanceField();

    // TODO think about separate function for initialization
    mask = Image(resized.size());
    canvas = Image(resized.size());
    debugCanvas = Image(resized.size());

    // cout << "contour pipeline complete" << endl;
    // cout << "strokes_image: " << strokes_image << endl;


}

void Mosaic::selectStroke(int stroke_id) { 

    if (stroke_id >= strokes.size()) { 
        return;
    }

    Color color(255);
    selected_stroke = Image(resized.size());
    Graphics::drawStroke(selected_stroke, strokes[stroke_id], color);
}















bool Mosaic::isValidTile(Point center, double size, double theta_deg) {

    if (mask.empty()) { 
        mask = Image(resized.size());
    }



    // check for validity
    if (theta_deg == ERROR_CODE_NO_VALID_THETA) { 
        return false;
    }
    if (!tileInBounds(center)) { 
        return false;
    }
    if (tileOverlapsMask(center, size, theta_deg)) { 
        return false;
    }
    return true;
}



bool Mosaic::tileInBounds(const Point& center) { 
    // Early exit if tile is far outside the image
    // int margin = static_cast<int>(2 * tileSize);
    int margin = 0;

    if (center.x < -margin || center.y < -margin ||
        center.x > mask.getWidth() + margin || center.y > mask.getHeight() + margin) {
        return false;
    }
    return true;
}

bool Mosaic::tileOverlapsMask(const Point& center, double tile_size, double theta_deg) {
    // Step 1: Early center check
    if (mask.at(center).r > 0) return true;

    double halfSize = tile_size / 2.0;
    double theta = theta_deg * M_PI / 180.0;
    Vec2d centerD(center);

    // Step 2: Define corners in local space
    std::vector<Vec2d> localCorners = {
        {-halfSize, -halfSize},
        { halfSize, -halfSize},
        { halfSize,  halfSize},
        {-halfSize,  halfSize}
    };

    // Step 2.1: Check rotated corners for early return
    for (const Vec2d& pt : localCorners) {
        double x = pt.x * std::cos(theta) - pt.y * std::sin(theta);
        double y = pt.x * std::sin(theta) + pt.y * std::cos(theta);
        Point worldPt(std::round(centerD.x + x), std::round(centerD.y + y));

        if (worldPt.x >= 0 && worldPt.x < mask.getWidth() &&
            worldPt.y >= 0 && worldPt.y < mask.getHeight()) {
            if (mask.at(worldPt).r > 0) {
                return true;
            }
        }
    }

    // Step 3: Sample along edges of the tile (skipping corners already checked)
    const int samplesPerEdge = 10;
    for (int i = 0; i < 4; ++i) {
        Vec2d start = localCorners[i];
        Vec2d end = localCorners[(i + 1) % 4];

        for (int s = 1; s < samplesPerEdge; ++s) { // skip s = 0 and s = samplesPerEdge
            double t = static_cast<double>(s) / samplesPerEdge;
            Vec2d ptLocal(
                start.x * (1 - t) + end.x * t,
                start.y * (1 - t) + end.y * t
            );

            double x = ptLocal.x * std::cos(theta) - ptLocal.y * std::sin(theta);
            double y = ptLocal.x * std::sin(theta) + ptLocal.y * std::cos(theta);
            Point worldPt(std::round(centerD.x + x), std::round(centerD.y + y));

            if (worldPt.x >= 0 && worldPt.x < mask.getWidth() &&
                worldPt.y >= 0 && worldPt.y < mask.getHeight()) {
                if (mask.at(worldPt).r > 0) {
                    return true;
                }
            }
        }
    }

    return false;
}





TileInfo Mosaic::placeTile(Point center, double size, double theta_deg, int frontier, string text) {



    Graphics::drawSquare(mask, center, size, theta_deg, Color(255), size);


    // TODO add tile metadata to the tiles_placed vector
    int order = tiles_placed.size();
    Color color = sampleTileColor(center, size, theta_deg);
    TileInfo current_tile = {
        center,
        size, 
        theta_deg,
        color,
        order,
        frontier
    };
    tiles_placed.push_back(current_tile);
    // tiles_to_render.push_back(current_tile);

    // if (tiles_to_render.size() >= params.tiles_per_frame) { 
    //     renderTiles();
    //     tiles_to_render.clear();
    // }

    return current_tile;

}
















Point Mosaic::getRandomPointOnStroke(int stroke_id) {
    // Safety check: make sure k is in bounds
    if (stroke_id < 0 || stroke_id >= static_cast<int>(strokes.size())) {
        throw std::out_of_range("Segment index k is out of range");
    }

    const std::vector<Point>& stroke = strokes.at(stroke_id);
    if (stroke.empty()) {
        throw std::runtime_error("No points in the selected segment");
    }

    return Random::selectFromVector<Point>(stroke);
}











double Mosaic::findBestTheta(Point center, double size) { 

    int radius = static_cast<int>(size * 0.69); // HUGE impact on alignment TODO do some geometry


    // Find non-zero stroke pixels
    std::vector<Point> region_pixels = findNonZeroInRadius(selected_stroke, center, radius);

    // cout << "region_pixels.size(): " << region_pixels.size() << endl;

    // Check if enough points for PCA
    if (region_pixels.size() < 2) {
        return ERROR_CODE_NO_VALID_THETA;
    }

    Vec2d direction = Geometry::pcaDirection(region_pixels);
    double theta_deg = Geometry::vectorToAngleDegrees(direction);


    return theta_deg;
}


std::vector<Point> Mosaic::findNonZeroInRadius(const Image& src, const Point& center, int radius) {
    std::vector<Point> result;

    int cx = center.x;
    int cy = center.y;
    int r2 = radius * radius;

    int xmin = std::max(0, cx - radius);
    int xmax = std::min(src.getWidth() - 1, cx + radius);
    int ymin = std::max(0, cy - radius);
    int ymax = std::min(src.getHeight() - 1, cy + radius);

    for (int y = ymin; y <= ymax; ++y) {
        for (int x = xmin; x <= xmax; ++x) {
            int dx = x - cx;
            int dy = y - cy;
            if (dx * dx + dy * dy <= r2) {
                const Color& c = src.at(Point(x, y));
                if (c.r > 0 || c.g > 0 || c.b > 0) {
                    result.emplace_back(x, y);
                }
            }
        }
    }

    return result;
}












std::vector<Point> Mosaic::findRingIntersections(const Point& center, double ring_size, double theta_deg, int thickness) {
    std::vector<Point> intersections;

    double halfSize = ring_size / 1.0; // TODO semantic clean up, intended behavior is distance from center
    double theta = theta_deg * M_PI / 180.0;
    Vec2d centerD(center);

    // Define corners in local space
    std::vector<Vec2d> localCorners = {
        {-halfSize, -halfSize},
        { halfSize, -halfSize},
        { halfSize,  halfSize},
        {-halfSize,  halfSize}
    };

    // DEBUG
    // canvas = selected_stroke.clone();

    for (int i = 0; i < 4; ++i) {
        Vec2d start = localCorners[i];
        Vec2d end   = localCorners[(i + 1) % 4];

        Vec2d edge = end - start;
        int numSteps = static_cast<int>(std::round(std::max(std::abs(edge.x), std::abs(edge.y))));
        if (numSteps == 0) continue;

        // Normalize edge direction and normal vector
        Vec2d edgeStep(edge.x / numSteps, edge.y / numSteps);
        Vec2d normal(-edgeStep.y, edgeStep.x); // 90° CCW

        for (int s = 0; s <= numSteps; ++s) {
            Vec2d ptLocal = start + edgeStep * s;

            for (int d = 0; d <= thickness; ++d) {
                Vec2d offset = ptLocal + normal * d;

                // Apply rotation
                double x = offset.x * std::cos(theta) - offset.y * std::sin(theta);
                double y = offset.x * std::sin(theta) + offset.y * std::cos(theta);
                Point worldPt(std::round(centerD.x + x), std::round(centerD.y + y));

                // // DEBUG
                // // set every pixel in mask as marked where we checked
                // Color color(108, 0, 210);
                // canvas.setPixel(worldPt.x, worldPt.y, color);

                if (worldPt.x >= 0 && worldPt.x < mask.getWidth() &&
                    worldPt.y >= 0 && worldPt.y < mask.getHeight()) {
                    if (selected_stroke.at(worldPt).r > 0) {
                        intersections.push_back(worldPt);
                    }
                }
            }
        }
    }

    // std::cout << "intersections.size(): " << intersections.size() << std::endl;


    // std::vector<Point> pixels_in_stroke;
    // for (int x = 0; x < selected_stroke.getWidth(); x++) { 
    //     for (int y = 0; y < selected_stroke.getHeight(); y++) { 
    //         Color pixel = selected_stroke.at(x, y);
    //         if (pixel.r > 0) { 
    //             pixels_in_stroke.emplace_back(Point(x, y));
    //         }
    //     }
    // }

    // cout << "pixels in stroke: " << pixels_in_stroke.size() << endl;






    return intersections;
}








std::vector<Point> Mosaic::findPointsMultipleRings(const Point& center, double theta_deg) { 

    std::vector<Point> all_intersections;


    double initial_size = params.initial_step; // TODO make this a param
    double thickness = 1; // TODO make this a param
    
    for (int i = 0; i < params.number_of_rings; i++) { 
        double current_ring_size = initial_size + i * params.step_size;
                
        std::vector<Point> current_ring_points = findRingIntersections(center, current_ring_size, theta_deg, thickness);


        all_intersections.insert(all_intersections.end(), current_ring_points.begin(), current_ring_points.end());

    }


    // filter sort and reverse
    int min_dist = params.min_intersection_distance;
    Geometry::filterUniquePoints(all_intersections, min_dist);
    std::sort(all_intersections.begin(), all_intersections.end(),
        [&center](const Point& a, const Point& b) {
            return Geometry::euclideanDistance(a, center) < Geometry::euclideanDistance(b, center);
        });
    std::reverse(all_intersections.begin(), all_intersections.end());

    return all_intersections;


}




void Mosaic::placeTilesAlongStroke(int stroke_id) { 

    selectStroke(stroke_id);

    Point center = getRandomPointOnStroke(stroke_id);     // TODO we can't just assume the first random spot will be valid for tile placement. (although this works first time)
    double size = params.tile_size;


    stack<Point> stack;
    Point current;
    stack.push(center);
    int squares_placed = 0;
    int frontier = 0;


    while(!stack.empty()) { 

        current = stack.top();
        stack.pop();

        double theta_deg = findBestTheta(current, size);
        if (!isValidTile(current, size, theta_deg)) { 
            continue;
        }
        
        placeTile(current, size, theta_deg, frontier, std::to_string(squares_placed));

        
        std::vector<Point> intersections = findPointsMultipleRings(current, theta_deg);

        for (Point p : intersections) { 
            stack.push(p);
        }

    }


}


void Mosaic::placeTilesAllStrokes() { 

    for (int i = 0; i < strokes.size(); i++) { 
        selectStroke(i);
        placeTilesAlongStroke(i);
    }
}












void Mosaic::computeDistanceField() { 
    
    std::vector<float> distance_field = image::process::computeDistanceField(canny);
    
    image::process::computeSobelGradients(distance_field, canny.size(), grad_x, grad_y);

}

double Mosaic::findThetaTangent(Point center) {

    if (grad_x.empty() || grad_y.empty()) { 
        computeDistanceField();
    }

    int width = canny.getWidth();
    int height = canny.getHeight();
        
    int x = center.x;
    int y = center.y;

    int idx = y * width + x;

    float gx = grad_x[idx];
    float gy = grad_y[idx];

    // Compute the normal vector (∇f = [gx, gy])
    float nx = gx;
    float ny = gy;

    // Tangent is perpendicular to gradient
    float tx = -ny;
    float ty = nx;

    // Normalize
    float len = std::sqrt(tx * tx + ty * ty);
    if (len > 1e-6f) {
        tx /= len;
        ty /= len;
    } else {
        tx = 0.0f;
        ty = 0.0f;
    }

    double theta_rad = std::atan2(ty, tx);
    double theta_deg = theta_rad * 180.0 / M_PI;

    return theta_deg;
}








void Mosaic::floodFill() {

    double distance_from_center = params.distance_from_center;
    const int max_frontiers = params.max_frontiers;
    int frontier = 1;
    std::vector<TileInfo> frontier_tiles(tiles_placed);

    while (!frontier_tiles.empty() && frontier <= max_frontiers) { 
        
        // Collect all flood fill points for this frontier
        std::vector<Point> all_flood_points;
        for (const TileInfo& tile : frontier_tiles) {

            int num_points = params.flood_fill_neighbor_points;
            int max_step = getJitter(frontier);
            std::vector<Point> points = Geometry::samplePointsSquareBorder(tile.center, tile.theta_deg, distance_from_center, num_points);
            std::vector<Point> jittered_points = Random::jitterPoints(points, max_step, mask.size());
            all_flood_points.insert(all_flood_points.end(), jittered_points.begin(), jittered_points.end());
        }

        // TODO sort somehow

        std::vector<TileInfo> next_frontier_tiles;
        // Now place tiles at unique positions
        for (const Point& pt : all_flood_points) {
            double theta_deg = findThetaTangent(pt);
            if (isValidTile(pt, params.tile_size, theta_deg)) { 
                TileInfo current_tile = placeTile(pt, params.tile_size, theta_deg, frontier + 1);
                next_frontier_tiles.push_back(current_tile);
            }
            
        }

        frontier_tiles = next_frontier_tiles;
        frontier++;
    }
}



int Mosaic::getJitter(int frontier) {
    // TODO fix this -- broken because of embind
    // if (params.jitter_map.empty()) { 
    //     return 0;
    // }
    // for (const auto& [threshold, jitter] : params.jitter_map) {
    //     if (frontier < threshold) {
    //         return jitter;
    //     }
    // }
    return 0;
}














void Mosaic::gapFill() { 
    if (grad_x.empty() || grad_y.empty()) { 
        computeDistanceField();
    }

    double tile_size = params.tile_size;
    int max_tiles_to_place = params.random_background_points;


    std::vector<Point> points;
    for (int y = 0; y < mask.getHeight(); ++y) {
        for (int x = 0; x < mask.getWidth(); ++x) {
            if (mask.at(x, y).r == 0) {
                points.push_back(Point(x, y));
            } 
        }
    }
    Random::shuffleVector(points);


   for (const Point& point : points) {


        // Sample guidance field
        
        double theta_deg = findThetaTangent(point);

        if (isValidTile(point, tile_size, theta_deg)) {

            int frontier = -1;
            placeTile(point, tile_size, theta_deg, frontier); 

        }
    }


}























void Mosaic::reconstructImage() { 

    // reset canvas
    canvas = Image(resized.size());

    for (TileInfo tile : tiles_placed) { 
        // Color color = sampleTileColor(tile);
        Graphics::drawSquare(canvas, tile.center, tile.size * 1.0, tile.theta_deg, tile.color, tile.size);
    }


}



void Mosaic::renderImageRange(int start, int num_tiles) { 

    int max_index = std::min<int>(tiles_placed.size(), start + num_tiles);

    for (int i = start; i < max_index; i++) { 
        TileInfo tile = tiles_placed[i];
        Graphics::drawSquare(canvas, tile.center, tile.size * 1.0, tile.theta_deg, tile.color, tile.size);
    }
    // render_pointer = start + num_tiles;
    if (debug_mode) { 
        renderDebugImageRange(start, num_tiles);
    }
}

void Mosaic::renderDebugImageRange(int start, int num_tiles) { 

    // cout << "rendering debug image" << endl;

    int max_index = std::min<int>(tiles_placed.size(), start + num_tiles);

    for (int i = start; i < max_index; i++) { 
        TileInfo tile = tiles_placed[i];
        Color color;
        if (tile.frontier == 0) { 
            color = Color(255, 0, 0);
            // cout << "using red" << endl;
        }
        else if (tile.frontier > 0) { 
            color = Color(0, 255, 0);
        }
        else if (tile.frontier == -1) { 
            color = Color(0, 100, 255);
        }
        else { 
            // cout << "tile stored incorrectly" << endl;
        }
        // cout << color << endl;
        Graphics::drawSquare(debugCanvas, tile.center, tile.size * 1.0, tile.theta_deg, color, tile.size);
    }
}


void Mosaic::setDebugMode(bool state) { 
    debug_mode = state;
}





int Mosaic::getRenderPointer() { 
    return std::min(static_cast<int>(tiles_placed.size()), render_pointer);
}

void Mosaic::setRenderPointer(int start) { 
    render_pointer = start;
}

void Mosaic::resetCanvas() { 
    canvas = Image(canvas.size());
    debugCanvas = Image(debugCanvas.size());
}


void Mosaic::setParameters(Parameters p) { 
    params = p;
}




void Mosaic::reconstructShowFrontiers() { 

    // reset canvas
    canvas = Image(resized.size());

    for (TileInfo tile : tiles_placed) { 
        // Color color = sampleTileColor(tile);
        Color color;
        if (tile.frontier == 0) { 
            color = Color(255, 0, 0);
        }
        else if (tile.frontier > 0) { 
            color = Color(0, 255, 0);
        }
        else if (tile.frontier == -1) {
            color = Color(0, 0, 255);
        }
        Graphics::drawSquare(canvas, tile.center, tile.size * 1.0, tile.theta_deg, color, tile.size);
    }


}






Color Mosaic::sampleTileColor(Point center, double size, double theta_deg) {

    size /= 2; // TODO make this tuneable

    std::vector<Point> points = Geometry::getPointsInsideSquare(center, size, theta_deg);

    // Accumulate RGB channels
    uint64_t r_sum = 0, g_sum = 0, b_sum = 0;
    int count = 0;

    for (const Point& pt : points) {
        if (resized.inBounds(pt.x, pt.y)) {
            Color sample = resized.at(pt);
            r_sum += sample.r;
            g_sum += sample.g;
            b_sum += sample.b;
            ++count;
        }
    }

    if (count == 0) return Color(0, 0, 0);  // Avoid division by zero

    uint8_t r_avg = static_cast<uint8_t>(r_sum / count);
    uint8_t g_avg = static_cast<uint8_t>(g_sum / count);
    uint8_t b_avg = static_cast<uint8_t>(b_sum / count);

    return Color(r_avg, g_avg, b_avg);
}






bool Mosaic::stepK(int k) { 
    for (int i = 0; i < k; i++) { 
        bool tile_placed = stepOnce();
        if (!tile_placed) {
            // cout << "step k exiting at i=" << i << endl;
            return false;
        }
    }
    return true;
}



bool Mosaic::stepOnce() { 

    while (true) {

        if (canny.empty()) {
            contourPipeline();
            if (canny.empty()) return false;
            continue;
        }

        int num_strokes = strokes.size();
        if (strokes_completed < num_strokes) { 

            if (strokePointsStack.empty()) { 
                int stroke_id = strokes_completed;
                selectStroke(stroke_id);
                int max_starts = 10;
                for (int i = 0; i < max_starts; i++) {
                    Point starting_point = getRandomPointOnStroke(stroke_id);
                    double starting_theta = findBestTheta(starting_point, params.tile_size);
                    if (isValidTile(starting_point, params.tile_size, starting_theta)) { 
                        int frontier = 0;
                        placeTile(starting_point, params.tile_size, starting_theta, frontier);
                        std::vector<Point> next_points = findPointsMultipleRings(starting_point, starting_theta);
                        for (Point p : next_points) { 
                            strokePointsStack.push(p);
                        }
                        return true;
                    }
                }
                // if we can't find a valid start, move on
                strokes_completed++;
                continue;
            }

            while (!strokePointsStack.empty()) { 
                Point pt = strokePointsStack.top();
                strokePointsStack.pop();
                double best_theta = findBestTheta(pt, params.tile_size);
                if (isValidTile(pt, params.tile_size, best_theta)) { 
                    int frontier = 0;
                    placeTile(pt, params.tile_size, best_theta, frontier);
                    std::vector<Point> next_points = findPointsMultipleRings(pt, best_theta);
                    for (Point p : next_points) { 
                        strokePointsStack.push(p);
                    }
                    return true;
                }
            }
            // exhausted points in stroke, move on
            strokes_completed++;
            continue;

        }

        int max_frontiers = params.max_frontiers;
        if (frontiers_completed < max_frontiers) { 

            if (floodPointsQueue.empty()) { 

                // get next set of points from last frontier
                for (TileInfo tile : tiles_placed) { 
                    if (tile.frontier == frontiers_completed) { 
                        double distance_from_center = params.distance_from_center;
                        int num_points = params.flood_fill_neighbor_points;
                        std::vector<Point> points = Geometry::samplePointsSquareBorder(tile.center, tile.theta_deg, distance_from_center, num_points);
                        for (Point pt : points) { 
                            floodPointsQueue.push(pt);
                        }
                    }
                }

                // if we didn't add any points, flood fill is done
                if (floodPointsQueue.empty()) { 
                    frontiers_completed = max_frontiers;
                }
                continue;
            }
            while (!floodPointsQueue.empty()) { 
                Point pt = floodPointsQueue.front();
                floodPointsQueue.pop();
                double best_theta = findThetaTangent(pt);
                if (isValidTile(pt, params.tile_size, best_theta)) { 
                    int frontier = frontiers_completed + 1;
                    placeTile(pt, params.tile_size, best_theta, frontier);
                    return true;
                }
            }
            // exhausted points in frontier, move on
            frontiers_completed++;
            continue;

        }

        if (gapPointsVector.empty()) { 
            if (!gaps_calculated) { 
                for (int y = 0; y < mask.getHeight(); ++y) {
                    for (int x = 0; x < mask.getWidth(); ++x) {
                        if (mask.at(x, y).r == 0) {
                            gapPointsVector.push_back(Point(x, y));
                        } 
                    }
                }
                Random::shuffleVector(gapPointsVector);
                gaps_calculated = true;
                continue;
            }
            else { 
                // gaps calculated, and all points checked -- we are done!
                return false;
            }

        }
        while (!gapPointsVector.empty()) { 
            Point pt = gapPointsVector.back();
            gapPointsVector.pop_back();
            double best_theta = findThetaTangent(pt);
            if (isValidTile(pt, params.tile_size, best_theta)) { 
                int frontier = -1;
                placeTile(pt, params.tile_size, best_theta, frontier);
                return true;
            }
        }
        // gap points exhausted -- we are done!
        return false;


    }

}






void Mosaic::clearData() { 
    /*
    Clear everything except for original (this shouldn't change ever)
    */

    params = Parameters();
       
    tiles_placed.clear();
    
    // image data various purposes
   
    resized = Image();
    canny = Image();
    strokes.clear();
    selected_stroke = Image();

    grad_x.clear();
    grad_y.clear();

    mask = Image();
    canvas = Image();
    debugCanvas = Image();

    std::stack<Point>().swap(strokePointsStack);
    std::queue<Point>().swap(floodPointsQueue);
    gapPointsVector.clear();

    strokes_completed = 0;
    frontiers_completed = 0;
    gaps_calculated = false;

    render_pointer = 0;



    // cout << "data cleared" << endl;
    // cout << "debug mode: " << debug_mode << endl;
}


void Mosaic::resizeOriginal() {
    Size target_size = original.size() * params.resize_factor;
    if (resized.size() != target_size) { 
        image::process::resize(original, resized, target_size);
        // cout << "resized image to: " << resized.size() << endl;
    }
}










}