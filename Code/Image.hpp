#pragma once

#include <cstdint>
#include <ostream>

namespace image { 

struct Color { 

    uint8_t r;
    uint8_t g;
    uint8_t b;

    Color() : r(0), g(0), b(0) {}

    Color(uint8_t value) 
        : r(value), g(value), b(value) {}
    Color(uint8_t red, uint8_t green, uint8_t blue) 
        : r(red), g(green), b(blue) {}

    Color(int value) 
        : r(clampToByte(value)), g(clampToByte(value)), b(clampToByte(value)) {}
    Color(int red, int green, int blue) 
        : r(clampToByte(red)), g(clampToByte(green)), b(clampToByte(blue)) {}

    Color(float value) 
        : r(clampToByte(static_cast<int>(value))), g(clampToByte(static_cast<int>(value))), b(clampToByte(static_cast<int>(value))) {}
    Color(float red, float green, float blue) 
        : r(clampToByte(static_cast<int>(red))), g(clampToByte(static_cast<int>(green))), b(clampToByte(static_cast<int>(blue))) {}

    Color(double value) 
        : r(clampToByte(static_cast<int>(value))), g(clampToByte(static_cast<int>(value))), b(clampToByte(static_cast<int>(value))) {}
    Color(double red, double green, double blue) 
        : r(clampToByte(static_cast<int>(red))), g(clampToByte(static_cast<int>(green))), b(clampToByte(static_cast<int>(blue))) {}
 

    bool operator==(const Color& other) const {
        return r == other.r && g == other.g && b == other.b;
    }

    bool operator!=(const Color& other) const {
        return !(*this == other);
    }
    

    private:

        static uint8_t clampToByte(int value) { 
            if (value < 0) return 0;
            else if (value > 255) return 255;
            return static_cast<uint8_t>(value);
        }

};



struct Point { 

    int x;
    int y;


    Point() : x(0), y(0) {}
    Point(int x, int y) : x(x), y(y) {}
    Point(float x, float y) : x(static_cast<int>(x)), y(static_cast<int>(y)) {}
    Point(double x, double y) : x(static_cast<int>(x)), y(static_cast<int>(y)) {}


    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Point& other) const {
        return !(*this == other);
    }



};


// Stream operator for Color
inline std::ostream& operator<<(std::ostream& os, const Color& color) {
    os << "(" << static_cast<int>(color.r) << ", "
                   << static_cast<int>(color.g) << ", "
                   << static_cast<int>(color.b);
    os << ")";
    return os;
}

// Stream operator for Point
inline std::ostream& operator<<(std::ostream& os, const Point& point) {
    os << "[" << point.x << ", " << point.y << "]";
    return os;
}



}