#include "imageProcess.hpp"

#include <queue>

namespace image::process { 




    Size resize(Image& src, Image& dest, int w, int h) {
        if (w <= 0 || h <= 0) {
            throw std::invalid_argument("resize: width and height must be positive");
        }

        Size newSize(w, h);
        dest = Image(newSize);  // Reallocate destination image with new size


        const double scaleX = static_cast<double>(src.getWidth()) / w;
        const double scaleY = static_cast<double>(src.getHeight()) / h;

        for (int y = 0; y < h; ++y) {
            int srcY = static_cast<int>(y * scaleY);
            if (srcY >= src.getHeight()) srcY = src.getHeight() - 1;

            for (int x = 0; x < w; ++x) {
                int srcX = static_cast<int>(x * scaleX);
                if (srcX >= src.getWidth()) srcX = src.getWidth() - 1;

                dest.setPixel(x, y, src.at(srcX, srcY));
            }
        }

        return newSize;
    }

    Size resize(Image& src, Image& dest, Size size) { 
        return resize(src, dest, size.height, size.width);
    }

    Size resize(Image& src, Image& dest, double ratio) { 
        int new_width = static_cast<int>(src.getWidth() * ratio);
        int new_height = static_cast<int>(src.getHeight() * ratio);

        return resize(src, dest, new_width, new_height);
    }




    void grayscale(Image& src, Image& dest) { 
        Size size = src.size();
        dest = Image(size);  // reallocate dest image with same size

        for (int y = 0; y < size.height; ++y) {
            for (int x = 0; x < size.width; ++x) {
                const Color& pixel = src.at(x, y);

                // Compute luminance (grayscale intensity)
                uint8_t gray = static_cast<uint8_t>(
                    0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b
                );
                Color new_pixel(gray, gray, gray);
                dest.setPixel(x, y, new_pixel);
            }
        }
    }















    void gaussianBlur(Image& src, Image& dest, Size kernel_size, double blur_sigma) {
        const int width = src.getWidth();
        const int height = src.getHeight();
    
        if (kernel_size.width % 2 == 0 || kernel_size.height % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd.");
        }
    
        const int radiusX = kernel_size.width / 2;
        const int radiusY = kernel_size.height / 2;
    
        std::vector<double> kernelX = generateGaussianKernel1D(radiusX, blur_sigma);
        std::vector<double> kernelY = generateGaussianKernel1D(radiusY, blur_sigma);
    
        Image temp(width, height);  // Intermediate buffer
    
        // Horizontal pass
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double r = 0, g = 0, b = 0, a = 0;
                for (int k = -radiusX; k <= radiusX; ++k) {
                    int sx = std::clamp(x + k, 0, width - 1);
                    Color sample = src.at(sx, y);
                    double w = kernelX[k + radiusX];
                    r += sample.r * w;
                    g += sample.g * w;
                    b += sample.b * w;
                    a += sample.a * w;
                }
                temp.at(x, y) = Color(
                    static_cast<uint8_t>(r),
                    static_cast<uint8_t>(g),
                    static_cast<uint8_t>(b),
                    static_cast<uint8_t>(a)
                );
            }
        }
    
        dest = Image(width, height);  // Allocate output
    
        // Vertical pass
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double r = 0, g = 0, b = 0, a = 0;
                for (int k = -radiusY; k <= radiusY; ++k) {
                    int sy = std::clamp(y + k, 0, height - 1);
                    Color sample = temp.at(x, sy);
                    double w = kernelY[k + radiusY];
                    r += sample.r * w;
                    g += sample.g * w;
                    b += sample.b * w;
                    a += sample.a * w;
                }
                dest.at(x, y) = Color(
                    static_cast<uint8_t>(r),
                    static_cast<uint8_t>(g),
                    static_cast<uint8_t>(b),
                    static_cast<uint8_t>(a)
                );
            }
        }
    }

    std::vector<double> generateGaussianKernel1D(int radius, double sigma) {
        int size = 2 * radius + 1;
        std::vector<double> kernel(size);
        double sum = 0.0;
    
        const double coeff = 1.0 / (std::sqrt(2.0 * M_PI) * sigma);
        const double denom = 2.0 * sigma * sigma;
    
        for (int i = -radius; i <= radius; ++i) {
            double value = coeff * std::exp(-(i * i) / denom);
            kernel[i + radius] = value;
            sum += value;
        }
    
        // Normalize kernel
        for (double& value : kernel) {
            value /= sum;
        }
    
        return kernel;
    }
    





    void sobelFilter(const Image& src, Image& dest_grad_x, Image& dest_grad_y) {
        int w = src.getWidth();
        int h = src.getHeight();
    
        dest_grad_x = Image(w, h);
        dest_grad_y = Image(w, h);
    
        // 3x3 Sobel kernels
        int kernelX[3][3] = {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };
    
        int kernelY[3][3] = {
            { -1, -2, -1 },
            {  0,  0,  0 },
            {  1,  2,  1 }
        };
    
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                int gx = 0, gy = 0;
    
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int pixel = src.at(x + kx, y + ky).r; // grayscale: use .r channel
                        gx += pixel * kernelX[ky + 1][kx + 1];
                        gy += pixel * kernelY[ky + 1][kx + 1];
                    }
                }
    
                uint8_t gxVal = static_cast<uint8_t>(std::clamp(gx + 128, 0, 255));
                uint8_t gyVal = static_cast<uint8_t>(std::clamp(gy + 128, 0, 255));
                dest_grad_x.setPixel(x, y, Color(gxVal, gxVal, gxVal));
                dest_grad_y.setPixel(x, y, Color(gyVal, gyVal, gyVal));
            }
        }
    }

    void visualizeSobel(const Image& gradX, const Image& gradY, Image& dest) {
        int width = gradX.getWidth();
        int height = gradX.getHeight();
        dest = Image(width, height);
    
        double maxMag = 0.0;
        std::vector<double> magnitudes(width * height);
    
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int gx = static_cast<int>(gradX.at(x, y).r) - 128;
                int gy = static_cast<int>(gradY.at(x, y).r) - 128;
    
                double mag = std::sqrt(gx * gx + gy * gy);
                magnitudes[y * width + x] = mag;
                if (mag > maxMag) maxMag = mag;
            }
        }
    
        // Avoid division by 0 in blank image
        if (maxMag < 1e-5) maxMag = 1.0;
    
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double mag = magnitudes[y * width + x];
                uint8_t val = static_cast<uint8_t>((mag / maxMag) * 255.0);
                dest.setPixel(x, y, Color(val, val, val));
            }
        }
    }
    











    
    
    

    

    void cannyFilter(Image& src_blurred, Image& dest, int canny_threshold_1, int canny_threshold_2) {
        int width = src_blurred.getWidth();
        int height = src_blurred.getHeight();
    
        std::vector<int> gradX, gradY;
        sobelFilterRaw(src_blurred, gradX, gradY);
    
        std::vector<double> gradMag(width * height);
        std::vector<double> gradDir(width * height);
    
        for (int i = 0; i < width * height; ++i) {
            gradMag[i] = std::sqrt(gradX[i] * gradX[i] + gradY[i] * gradY[i]);
            gradDir[i] = std::atan2(gradY[i], gradX[i]);
        }
    
        // -- Non-Max Suppression (same as before but now uses better gradients)
        std::vector<uint8_t> nms(width * height, 0);
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int i = y * width + x;
                double angle = gradDir[i];
                double mag = gradMag[i];
        
                // Gradient direction components
                double dx = std::cos(angle);
                double dy = std::sin(angle);
        
                // Interpolate two magnitudes along gradient direction
                double mag1, mag2;
        
                // Point in positive gradient direction
                double x1 = x + dx;
                double y1 = y + dy;
        
                // Point in negative gradient direction
                double x2 = x - dx;
                double y2 = y - dy;
        
                auto getInterpolated = [&](double x, double y) -> double {
                    int x0 = int(x);
                    int y0 = int(y);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
        
                    double dx = x - x0;
                    double dy = y - y0;
        
                    // Bilinear interpolation
                    double m00 = gradMag[y0 * width + x0];
                    double m10 = gradMag[y0 * width + x1];
                    double m01 = gradMag[y1 * width + x0];
                    double m11 = gradMag[y1 * width + x1];
        
                    return (1 - dx) * (1 - dy) * m00 +
                           dx * (1 - dy) * m10 +
                           (1 - dx) * dy * m01 +
                           dx * dy * m11;
                };
        
                mag1 = getInterpolated(x1, y1);
                mag2 = getInterpolated(x2, y2);
        
                if (mag >= mag1 && mag >= mag2) {
                    nms[i] = static_cast<uint8_t>(std::clamp(mag, 0.0, 255.0));
                } else {
                    nms[i] = 0;
                }
            }
        }
        
    
        // -- Double Thresholding
        std::vector<uint8_t> edgeMap(width * height, 0);
        for (int i = 0; i < width * height; ++i) {
            if (nms[i] >= canny_threshold_2) edgeMap[i] = 255;
            else if (nms[i] >= canny_threshold_1) edgeMap[i] = 128;
        }
    
        // -- Edge tracking (non-recursive version)
        std::queue<std::pair<int, int>> q;
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int i = y * width + x;
                if (edgeMap[i] == 255) {
                    q.push({x, y});
                }
            }
        }
    
        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();
    
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                    int ni = ny * width + nx;
                    if (edgeMap[ni] == 128) {
                        edgeMap[ni] = 255;
                        q.push({nx, ny});
                    }
                }
            }
        }
    
        // -- Final output
        dest = Image(width, height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t val = (edgeMap[y * width + x] == 255) ? 255 : 0;
                dest.setPixel(x, y, Color(val, val, val));
            }
        }
    }

    // TODO unify sobel filter functions
    void sobelFilterRaw(const Image& src, std::vector<int>& gradX, std::vector<int>& gradY) {
        int w = src.getWidth();
        int h = src.getHeight();
        gradX.resize(w * h);
        gradY.resize(w * h);
    
        int kernelX[3][3] = {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };
    
        int kernelY[3][3] = {
            { -1, -2, -1 },
            {  0,  0,  0 },
            {  1,  2,  1 }
        };
    
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                int gx = 0, gy = 0;
    
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int pixel = src.at(x + kx, y + ky).r;
                        gx += pixel * kernelX[ky + 1][kx + 1];
                        gy += pixel * kernelY[ky + 1][kx + 1];
                    }
                }
    
                int idx = y * w + x;
                gradX[idx] = gx;
                gradY[idx] = gy;
            }
        }
    }
    
     








    void findContours(const Image& src_binary, std::vector<std::vector<Point>>& contours) {
        int width = src_binary.getWidth();
        int height = src_binary.getHeight();
    
        std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
        contours.clear();
    
        auto isValid = [&](int x, int y) {
            return x >= 0 && y >= 0 && x < width && y < height;
        };
    
        std::vector<Point> directions = {
            {-1, -1}, {0, -1}, {1, -1},
            {-1,  0},          {1,  0},
            {-1,  1}, {0,  1}, {1,  1}
        };
    
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (visited[y][x]) continue;
    
                uint8_t val = src_binary.at(x, y).r;  // assume grayscale (R=G=B)
                if (val == 0) continue;
    
                // Start new contour
                std::vector<Point> contour;
                std::vector<Point> stack = { Point(x, y) };
    
                visited[y][x] = true;
    
                while (!stack.empty()) {
                    Point p = stack.back();
                    stack.pop_back();
    
                    contour.push_back(p);
    
                    for (const auto& d : directions) {
                        int nx = p.x + d.x;
                        int ny = p.y + d.y;
    
                        if (!isValid(nx, ny)) continue;
                        if (visited[ny][nx]) continue;
                        if (src_binary.at(nx, ny).r == 0) continue;
    
                        stack.emplace_back(nx, ny);
                        visited[ny][nx] = true;
                    }
                }
    
                if (!contour.empty()) {
                    contours.push_back(std::move(contour));
                }
            }
        }
    }




    
    











}