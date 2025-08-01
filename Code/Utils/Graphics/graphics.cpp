#include "graphics.hpp"
#include <iostream>
#include <cmath>
#include <vector>

#include "../Random/random.hpp"

using namespace std;

namespace Graphics { 


    void drawLine(Image& image, const Point& point_a, const Point& point_b, int thickness, const Color& color) {
        if (thickness <= 0) return;
    
        int x0 = point_a.x;
        int y0 = point_a.y;
        int x1 = point_b.x;
        int y1 = point_b.y;
    
        int dx = std::abs(x1 - x0);
        int dy = std::abs(y1 - y0);
    
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;
    
        auto drawThickPixel = [&](int x, int y) {
            int r = thickness / 2;
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    int px = x + dx;
                    int py = y + dy;
                    if (px >= 0 && px < image.getWidth() && py >= 0 && py < image.getHeight()) {
                        image.setPixel(px, py, color);
                    }
                }
            }
        };
    
        while (true) {
            drawThickPixel(x0, y0);
            if (x0 == x1 && y0 == y1) break;
    
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx)  { err += dx; y0 += sy; }
        }
    }


    void drawArrow(Image& image, const Point& center, int length, int thickness, double angle_deg, const Color& color) {
        if (image.empty()) {
            std::cerr << "drawArrow: Input image is empty." << std::endl;
            return;
        }
    
        // Parameters for arrow size (in pixels)

        const int headLength = static_cast<int>(length / 3);        // length of each arrowhead wing
        const double headAngleDeg = 30;  // angle between shaft and arrowhead wing

    
        // Convert angle to radians
        double theta = angle_deg * M_PI / 180.0;
    
        // Compute arrow tip point
        Point tip(
            std::round(center.x + length * cos(theta)),
            std::round(center.y + length * sin(theta))
        );
    
        // Draw shaft line
        drawLine(image, center, tip, thickness, color);
    
        // Calculate left wing point
        double leftTheta = theta + (M_PI * headAngleDeg / 180.0);
        Point leftWing(
            std::round(tip.x - headLength * cos(leftTheta)),
            std::round(tip.y - headLength * sin(leftTheta))
        );
    
        // Calculate right wing point
        double rightTheta = theta - (M_PI * headAngleDeg / 180.0);
        Point rightWing(
            std::round(tip.x - headLength * cos(rightTheta)),
            std::round(tip.y - headLength * sin(rightTheta))
        );
    
        // Draw arrowhead wings
        drawLine(image, tip, leftWing, thickness, color);
        drawLine(image, tip, rightWing, thickness, color);
    }
    


    void drawFilledPolygon(Image& image, const std::vector<Point>& polygon, const Color& color) {
        if (polygon.size() < 3) return;
    
        // Find vertical bounds of the polygon
        int minY = polygon[0].y, maxY = polygon[0].y;
        for (const auto& p : polygon) {
            minY = std::min(minY, p.y);
            maxY = std::max(maxY, p.y);
        }
    
        // Clip to image height
        minY = std::max(minY, 0);
        maxY = std::min(maxY, image.getHeight() - 1);
    
        // Scanline fill
        for (int y = minY; y <= maxY; ++y) {
            std::vector<int> intersections;
    
            for (size_t i = 0; i < polygon.size(); ++i) {
                const Point& p1 = polygon[i];
                const Point& p2 = polygon[(i + 1) % polygon.size()];
    
                // Check if scanline intersects the edge
                if ((p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y)) {
                    // Compute intersection x coordinate
                    float x = p1.x + (float)(y - p1.y) * (p2.x - p1.x) / (float)(p2.y - p1.y);
                    intersections.push_back(static_cast<int>(std::round(x)));
                }
            }
    
            // Sort x intersections
            std::sort(intersections.begin(), intersections.end());
    
            // Fill between pairs
            for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
                int x0 = intersections[i];
                int x1 = intersections[i + 1];
                if (x1 < x0) std::swap(x0, x1);
    
                x0 = std::max(x0, 0);
                x1 = std::min(x1, image.getWidth() - 1);
    
                for (int x = x0; x <= x1; ++x) {
                    image.setPixel(x, y, color);
                }
            }
        }
    }



    void drawSquare(Image& image, const Point& center, double size, double angle_deg, const Color& color, int border_width) {
        if (size <= 0 || border_width < 0) {
            std::cerr << "drawSquare: Invalid size or border width.\n";
            return;
        }
    
        double half_size = size / 2.0;
    
        // If the border width consumes the entire square, draw it filled
        if (2 * border_width >= size) {
            // Build outer square corners in local space
            std::vector<Point> outer = {
                {-half_size, -half_size},
                { half_size, -half_size},
                { half_size,  half_size},
                {-half_size,  half_size}
            };
    
            double angle_rad = angle_deg * M_PI / 180.0;
            double cos_theta = std::cos(angle_rad);
            double sin_theta = std::sin(angle_rad);
    
            std::vector<Point> rotated;
            for (const auto& pt : outer) {
                double x = pt.x * cos_theta - pt.y * sin_theta;
                double y = pt.x * sin_theta + pt.y * cos_theta;
                rotated.emplace_back(center.x + static_cast<int>(std::round(x)),
                                     center.y + static_cast<int>(std::round(y)));
            }
    
            drawFilledPolygon(image, rotated, color);
            return;
        }
    
        // Otherwise, construct the outer and inner rotated squares
        double inner_half = half_size - border_width;
    
        std::vector<Point> outer = {
            {-half_size, -half_size},
            { half_size, -half_size},
            { half_size,  half_size},
            {-half_size,  half_size}
        };
    
        std::vector<Point> inner = {
            {-inner_half, -inner_half},
            { inner_half, -inner_half},
            { inner_half,  inner_half},
            {-inner_half,  inner_half}
        };
    
        double angle_rad = angle_deg * M_PI / 180.0;
        double cos_theta = std::cos(angle_rad);
        double sin_theta = std::sin(angle_rad);
    
        std::vector<Point> outer_rotated, inner_rotated;
        for (int i = 0; i < 4; ++i) {
            double ox = outer[i].x * cos_theta - outer[i].y * sin_theta;
            double oy = outer[i].x * sin_theta + outer[i].y * cos_theta;
            outer_rotated.emplace_back(center.x + static_cast<int>(std::round(ox)),
                                       center.y + static_cast<int>(std::round(oy)));
    
            double ix = inner[i].x * cos_theta - inner[i].y * sin_theta;
            double iy = inner[i].x * sin_theta + inner[i].y * cos_theta;
            inner_rotated.emplace_back(center.x + static_cast<int>(std::round(ix)),
                                       center.y + static_cast<int>(std::round(iy)));
        }
    
        // Draw 4 border trapezoids between outer and inner square
        for (int i = 0; i < 4; ++i) {
            int next = (i + 1) % 4;
            std::vector<Point> trapezoid = {
                outer_rotated[i],
                outer_rotated[next],
                inner_rotated[next],
                inner_rotated[i]
            };
            drawFilledPolygon(image, trapezoid, color);
        }
    }





    void drawStroke(Image& image, const std::vector<Point>& stroke, const Color& color) {
        for (const Point& pt : stroke) {
            if (pt.y >= 0 && pt.y < image.getHeight() && pt.x >= 0 && pt.x < image.getWidth()) {
                image.setPixel(pt.x, pt.y, color);
            }
        }
    }


    void drawStrokesRandomColor(Image& image, const std::vector<std::vector<Point>>& strokes) { 

        std::vector<image::Color> colors_used;
        

        for (std::vector<Point> stroke : strokes) { 
            Color color;
            do {
                color = Random::randomColor();
            } while (std::find(colors_used.begin(), colors_used.end(), color) != colors_used.end());
            
            colors_used.push_back(color);
            
            for (const auto& pt : stroke) {
                if (pt.y >= 0 && pt.y < image.getHeight() && pt.x >= 0 && pt.x < image.getWidth()) {
                    image.setPixel(pt.x, pt.y, color);
                }
            }
        }
    }
    

}