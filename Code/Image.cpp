#include "Image.hpp"

namespace image { 










// Default constructor
Image::Image() : width_(0), height_(0) {}

// Param constructors
Image::Image(int width, int height, const Color&fill) 
    : width_(width), height_(height), data_(std::vector<Color>(width * height, fill)) {}

// Copy constructor
Image::Image(const Image& other) 
    : width_(other.width_), height_(other.height_), data_(other.data_) {}

// Move constructor
Image::Image(Image&& other) noexcept 
    : width_(other.width_), height_(other.height_)  {
    data_ = std::move(other.data_);
    other.width_ = 0;
    other.height_ = 0;
}

// Copy assignment
Image& Image::operator=(const Image& other) {
    if (this != &other) { 
        width_ = other.width_;
        height_ = other.height_;
        data_ = other.data_;
    }
    return *this;
}

// Move assignment
Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) { 
        width_ = other.width_;
        height_ = other.height_;
        data_ = std::move(other.data_);

        other.width_ = 0;
        other.height_ = 0;
    }
    return *this;
}

// Destructor
Image::~Image() {}


Color& Image::at(int x, int y) {
    return data_[y * width_ + x];
}
const Color& Image::at(int x, int y) const {
    return data_[y * width_ + x];
}

Size Image::size() const { 
    return Size(width_, height_);
}





}