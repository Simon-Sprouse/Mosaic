#pragma once

#include "../Image/Image.hpp"

// #include <opencv2/opencv.hpp>
#include <string>


namespace Graphics { 

    using image::Image;
    using image::Point;
    using image::Color;



    void drawLine(Image& image, const Point& point_a, const Point& point_b, int thickness, const Color& color);
    void drawArrow(Image& image, const Point& center, int length, int thickness, double angle_deg, const Color& color);

    void drawFilledPolygon(Image& image, const std::vector<Point>& polygon, const Color& color);
    void drawSquare(Image& image, const Point& center, double size, double angle_deg, const Color& color, int border_width);

    void drawStroke(Image& image, const std::vector<Point>& strokes, const Color& color);
    void drawStrokesRandomColor(Image& image, const std::vector<std::vector<Point>>& strokes);


    // TODO this will take some time
    // void drawText(Image& image, const Point& center, double size, double angle_deg, const Color& color);



}

