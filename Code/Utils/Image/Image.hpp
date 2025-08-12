#pragma once

#include <stdint.h>
#include <ostream>
#include <vector>

namespace image { 

struct Color { 

    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;

    Color() : r(0), g(0), b(0), a(255) {}

    explicit Color(uint8_t value) 
        : r(value), g(value), b(value), a(255) {}
    Color(uint8_t red, uint8_t green, uint8_t blue) 
        : r(red), g(green), b(blue), a(255) {}
    Color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha) 
        : r(red), g(green), b(blue), a(alpha) {}

    explicit Color(int value) 
        : r(clampToByte(value)), g(clampToByte(value)), b(clampToByte(value)), a(255) {}
    Color(int red, int green, int blue) 
        : r(clampToByte(red)), g(clampToByte(green)), b(clampToByte(blue)), a(255) {}
    Color(int red, int green, int blue, int alpha) 
        : r(clampToByte(red)), g(clampToByte(green)), b(clampToByte(blue)), a(clampToByte(alpha)) {}

    explicit Color(float value) 
        : r(clampToByte(static_cast<int>(value))), g(clampToByte(static_cast<int>(value))), 
          b(clampToByte(static_cast<int>(value))), a(255) {}
    Color(float red, float green, float blue) 
        : r(clampToByte(static_cast<int>(red))), g(clampToByte(static_cast<int>(green))), 
          b(clampToByte(static_cast<int>(blue))), a(255) {}
    Color(float red, float green, float blue, float alpha) 
        : r(clampToByte(static_cast<int>(red))), g(clampToByte(static_cast<int>(green))), 
          b(clampToByte(static_cast<int>(blue))), a(clampToByte(static_cast<int>(alpha))) {}

    explicit Color(double value) 
        : r(clampToByte(static_cast<int>(value))), g(clampToByte(static_cast<int>(value))), 
          b(clampToByte(static_cast<int>(value))), a(255) {}
    Color(double red, double green, double blue) 
        : r(clampToByte(static_cast<int>(red))), g(clampToByte(static_cast<int>(green))), 
          b(clampToByte(static_cast<int>(blue))), a(255) {}
    Color(double red, double green, double blue, double alpha) 
        : r(clampToByte(static_cast<int>(red))), g(clampToByte(static_cast<int>(green))), 
          b(clampToByte(static_cast<int>(blue))), a(clampToByte(static_cast<int>(alpha))) {}
 

    bool operator==(const Color& other) const {
        return r == other.r && g == other.g && b == other.b && a == other.a;
    }

    bool operator!=(const Color& other) const {
        return !(*this == other);
    }
    

    private:

        static constexpr uint8_t clampToByte(int value) { 
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

// Subtraction operator
inline Point operator-(const Point& a, const Point& b) {
    return Point(a.x - b.x, a.y - b.y);
}



struct Vec2d {
    double x;
    double y;

    Vec2d() : x(0), y(0) {}
    Vec2d(double x_, double y_) : x(x_), y(y_) {}

    // Constructor from Point
    Vec2d(const Point& p) : x(static_cast<double>(p.x)), y(static_cast<double>(p.y)) {}

    double dot(const Vec2d& other) const {
        return x * other.x + y * other.y;
    };


};

// Subtraction operator
inline Vec2d operator+(const Vec2d& a, const Vec2d& b) {
    return Vec2d(a.x + b.x, a.y + b.y);
}

inline Vec2d operator-(const Vec2d& a, const Vec2d& b) {
    return Vec2d(a.x - b.x, a.y - b.y);
}

// TODO finish all arithmetic
inline Vec2d operator*(const Vec2d& v, double scalar) {
    return Vec2d(v.x * scalar, v.y * scalar);
}

inline Vec2d operator/(const Vec2d& v, double scalar) {
    return Vec2d(v.x / scalar, v.y / scalar);
}




struct Size { 

    int width;
    int height;


    Size() : width(0), height(0) {}
    Size(int w, int h) : width(w), height(h) {}

    bool operator==(const Size& other) const {
        return width == other.width && height == other.height;
    }

    bool operator!=(const Size& other) const {
        return !(*this == other);
    }

    std::string toString() const;
    

};

inline Size operator*(const Size& s, double scalar) {
    return Size(s.width * scalar, s.height * scalar);
}



class Image {


    public: 


    // Default constructor
    Image();

    // Param constructors
    Image(int width, int height);
    Image(int width, int height, const Color& fill);
    Image(Size size);
    Image(Size size, const Color& fill);
    Image(Size size, const std::vector<float>& vec);

    // Copy constructor
    Image(const Image& other);

    // Move constructor
    Image(Image&& other) noexcept;

    // Copy assignment
    Image& operator=(const Image& other);

    // Move assignment
    Image& operator=(Image&& other) noexcept;

    // Destructor
    ~Image();

    Color& at(int x, int y);
    const Color& at(int x, int y) const;
    Color& at(Point pt);
    const Color& at(Point pt) const;
    Size size() const;
    int getHeight() const;
    int getWidth() const;
    uint8_t* rawData();
    const uint8_t* rawData() const;
    void setPixel(int x, int y, const Color& color);
    bool empty() const;
    void fill(const Color& color);
    Image clone() const;
    bool inBounds(int x, int y) const;

    bool operator==(const Image& other) const;
    bool operator!=(const Image& other) const;

    


    private:

        int getLinearIndex_(int x, int y) const;


        int width_;
        int height_;
        std::vector<Color> data_;



};






// Stream operator for Size
inline std::ostream& operator<<(std::ostream& os, const Size& s) {
    return os << "size[" << s.width << " x " << s.height << "]";
}

inline std::string Size::toString() const {
    return std::to_string(width) + "," + std::to_string(height);
}

// Stream operator for Color
inline std::ostream& operator<<(std::ostream& os, const Color& color) {

    os << "rgba[" << static_cast<int>(color.r) << ", "
                << static_cast<int>(color.g) << ", "
                << static_cast<int>(color.b) << ", "
                << static_cast<int>(color.a) << "]";


    return os;
}

// Stream operator for Point
inline std::ostream& operator<<(std::ostream& os, const Point& point) {
    os << "point[" << point.x << ", " << point.y << "]";
    return os;
}

// Stream operator for Point
inline std::ostream& operator<<(std::ostream& os, const Vec2d& vec) {
    os << "vec2d[" << vec.x << ", " << vec.y << "]";
    return os;
}


// Stream operator for Image
inline std::ostream& operator<<(std::ostream& os, const Image& image) {
    os << "image[" << image.getWidth() << " x " << image.getHeight() << "]";
    return os;
}







// std::vector<uint8_t> loadFile(const std::string& path) {
//     std::ifstream f(path, std::ios::binary);
//     return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
//                                 std::istreambuf_iterator<char>());
// }



Image fromEncodedBuffer(const uint8_t* data, size_t size);



}