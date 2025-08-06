#include "Image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // Put this in one .cpp file only!

namespace image { 






// Default constructor
Image::Image() : width_(0), height_(0) {}

// Param constructors
Image::Image(int width, int height)
    : width_(width), height_(height), data_(std::vector<Color>(width_* height_, Color())) {}
Image::Image(int width, int height, const Color& fill)
    : width_(width), height_(height), data_(std::vector<Color>(width_* height_, fill)) {}
Image::Image(Size size)
    : width_(size.width), height_(size.height), data_(std::vector<Color>(size.width * size.height, Color())) {}
Image::Image(Size size, const Color& fill)
    : width_(size.width), height_(size.height), data_(std::vector<Color>(size.width * size.height, fill)) {}

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
        data_.clear();

        width_ = other.width_;
        height_ = other.height_;
        data_ = other.data_;
    }
    return *this;
}

// Move assignment

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) { 
        data_.clear();

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





int Image::getLinearIndex_(int x, int y) const { 
    return y * width_ + x;
}


Color& Image::at(int x, int y) {
    return data_[getLinearIndex_(x, y)];
}
const Color& Image::at(int x, int y) const {
    return data_[getLinearIndex_(x, y)];
}
Color& Image::at(Point pt) {
    return data_[getLinearIndex_(pt.x, pt.y)];
}
const Color& Image::at(Point pt) const {
    return data_[getLinearIndex_(pt.x, pt.y)];
}

Size Image::size() const { 
    return Size(width_, height_);
}

int Image::getHeight() const {
    return height_;
}

int Image::getWidth() const {
    return width_;
}


bool Image::operator==(const Image& other) const {
    if (this->size() != other.size()) return false;

    for (int y = 0; y < getHeight(); ++y) {
        for (int x = 0; x < getWidth(); ++x) {
            if (this->at(x, y) != other.at(x, y))
                return false;
        }
    }
    return true;
}


bool Image::operator!=(const Image& other) const {
    return !(*this==other);
}



uint8_t* Image::rawData() {
    return reinterpret_cast<uint8_t*>(data_.data());
}

const uint8_t* Image::rawData() const {
    return reinterpret_cast<const uint8_t*>(data_.data());
}


void Image::setPixel(int x, int y, const Color& color) {
    data_.at(getLinearIndex_(x, y)) = color;
}


bool Image::empty() const { 
    return data_.empty();
}

void Image::fill(const Color& color) {
    std::fill(data_.begin(), data_.end(), color);
}


Image Image::clone() const {
    Image copy(width_, height_);
    copy.data_ = data_; // std::vector assignment performs a deep copy
    return copy;
}


bool Image::inBounds(int x, int y) const {
    return x >= 0 && y >= 0 && x < getWidth() && y < getHeight();
}



// TODO maybe this should be a constructor? 
Image fromEncodedBuffer(const uint8_t* data, size_t size) {
    int width, height, channels;

    unsigned char* pixels = stbi_load_from_memory(data, size, &width, &height, &channels, 4); // Force RGBA

    if (!pixels) {
        throw std::runtime_error("Failed to load image from memory");
    }

    Image img(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 4;
            Color c;
            c.r = pixels[idx + 0];
            c.g = pixels[idx + 1];
            c.b = pixels[idx + 2];
            img.setPixel(x, y, c);
        }
    }

    stbi_image_free(pixels);
    return img;
}




}