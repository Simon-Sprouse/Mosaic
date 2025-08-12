#include "imageProcess.hpp"
#include "../Geometry/geometry.hpp"

#include <queue>
#include <algorithm>

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
        return resize(src, dest, size.width, size.height);
    }

    Size resize(Image& src, Image& dest, double ratio) { 
        int new_width = static_cast<int>(src.getWidth() * ratio);
        int new_height = static_cast<int>(src.getHeight() * ratio);

        return resize(src, dest, new_width, new_height);
    }




    void grayscale(const Image& src, Image& dest) { 
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











    void gaussianBlur(Image& src, Image& dest, int kernel_size, double blur_sigma) {
        Size kernel(kernel_size, kernel_size);
        gaussianBlur(src, dest, kernel, blur_sigma);
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




        // TODO move image processing pipeline to new class
    int  divideIntoStrokes(const std::vector<std::vector<Point>>& cv_contours, 
                            std::vector<std::vector<Point>>& segment_points, 
                            Size image_size, 
                            int segment_angle_window, 
                            double max_segment_angle_rad, 
                            int min_segment_length) 
        { 

        // // Find contours
        // std::vector<std::vector<cv::Point>> cv_contours;
        // cv::findContours(edges.clone(), cv_contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        // Create an output color image
        Image contours(image_size);
        int contour_id = 0;



    

        for (const std::vector<Point>& contour : cv_contours) {
            if (contour.size() < 3)
                continue;

            std::vector<int> breaks;
            int len = contour.size();
            int w = segment_angle_window;

            for (int i = w; i < len - w; ++i) {
                Vec2d v1 = contour[i] - contour[i - w];
                Vec2d v2 = contour[i + w] - contour[i];

                double norm1 = std::sqrt(v1.x * v1.x + v1.y * v1.y) + 1e-8;
                double norm2 = std::sqrt(v2.x * v2.x + v2.y * v2.y) + 1e-8;

                Vec2d n1 = v1 / norm1;
                Vec2d n2 = v2 / norm2;

                double cosine = std::clamp(n1.dot(n2), -1.0, 1.0);
                double angle = std::acos(std::abs(cosine));

                if (angle > max_segment_angle_rad) {
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
                if (b - a < min_segment_length)
                    continue;



                segment_points.emplace_back(std::vector<Point>());
                for (int j = a; j < b; ++j) {
                    const Point& point = contour[j];
                    if (point.y >= 0 && point.y < image_size.height && point.x >= 0 && point.x < image_size.width) {
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



    // vvv OLD CODE vvv
    // void Mosaic::computeDistanceField() { 

    //     distance_map = cv::Mat::zeros(contours.size(), CV_8UC3);
    //     gradX = cv::Mat::zeros(contours.size(), CV_8UC3);
    //     gradY = cv::Mat::zeros(contours.size(), CV_8UC3);


    //     // Step 1: Convert to grayscale and binary edge map
    //     cv::Mat gray, binary;
    //     cv::cvtColor(contours, gray, cv::COLOR_BGR2GRAY);
    //     cv::threshold(gray, binary, 1, 255, cv::THRESH_BINARY);

    //     // Step 2: Invert binary
    //     cv::Mat inverted = 255 - binary;

    //     // Step 3: Compute distance transform
    //     cv::distanceTransform(inverted, distance_map, cv::DIST_L2, 3);

    //     // Step 4: Compute gradients
    //     cv::Sobel(distance_map, gradX, CV_32F, 1, 0, 3);
    //     cv::Sobel(distance_map, gradY, CV_32F, 0, 1, 3);


    // }


    // // TODO complete this code
    // void computeDistaceField(const Image& strokes_img_source, Image& grad_x_dest, Image& grad_y_dest) { 

    // }



    std::vector<float> computeDistanceField(const Image& strokes_img_source) {
        int width = strokes_img_source.getWidth();
        int height = strokes_img_source.getHeight();
    
        // Step 1: Convert to grayscale
        Image gray(width, height);
        grayscale(strokes_img_source, gray);
    
        // Step 2: Create binary map (threshold at 1, like OpenCV code)
        std::vector<bool> binary(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Threshold: anything > 1 becomes true (white), else false (black)
                binary[y * width + x] = (gray.at(x, y).r > 1);
            }
        }
    
        // Step 3: Initialize distance buffer
        std::vector<float> distance_field(width * height);
        
        // Set initial distances:
        // - 0 distance for edge pixels (binary = true, i.e., stroke pixels)
        // - max distance for background pixels (binary = false)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (binary[idx]) {
                    // This is a stroke/edge pixel - distance = 0
                    distance_field[idx] = 0.0f;
                } else {
                    // This is background - start with max distance
                    distance_field[idx] = std::numeric_limits<float>::max();
                }
            }
        }
    
        // Step 4: Two-pass chamfer distance transform
        // First pass: top-left to bottom-right
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                float& current_dist = distance_field[idx];
                
                // Skip if this is already a stroke pixel (distance = 0)
                if (current_dist == 0.0f) continue;
                
                // Check neighbors that have already been processed
                // Left neighbor
                if (x > 0) {
                    current_dist = std::min(current_dist, distance_field[y * width + (x - 1)] + 1.0f);
                }
                
                // Top neighbor
                if (y > 0) {
                    current_dist = std::min(current_dist, distance_field[(y - 1) * width + x] + 1.0f);
                }
                
                // Top-left diagonal
                if (x > 0 && y > 0) {
                    current_dist = std::min(current_dist, distance_field[(y - 1) * width + (x - 1)] + std::sqrt(2.0f));
                }
                
                // Top-right diagonal
                if (x < width - 1 && y > 0) {
                    current_dist = std::min(current_dist, distance_field[(y - 1) * width + (x + 1)] + std::sqrt(2.0f));
                }
            }
        }
    
        // Second pass: bottom-right to top-left
        for (int y = height - 1; y >= 0; --y) {
            for (int x = width - 1; x >= 0; --x) {
                int idx = y * width + x;
                float& current_dist = distance_field[idx];
                
                // Skip if this is already a stroke pixel (distance = 0)
                if (current_dist == 0.0f) continue;
                
                // Check neighbors that will be processed
                // Right neighbor
                if (x < width - 1) {
                    current_dist = std::min(current_dist, distance_field[y * width + (x + 1)] + 1.0f);
                }
                
                // Bottom neighbor
                if (y < height - 1) {
                    current_dist = std::min(current_dist, distance_field[(y + 1) * width + x] + 1.0f);
                }
                
                // Bottom-right diagonal
                if (x < width - 1 && y < height - 1) {
                    current_dist = std::min(current_dist, distance_field[(y + 1) * width + (x + 1)] + std::sqrt(2.0f));
                }
                
                // Bottom-left diagonal
                if (x > 0 && y < height - 1) {
                    current_dist = std::min(current_dist, distance_field[(y + 1) * width + (x - 1)] + std::sqrt(2.0f));
                }
            }
        }

        return distance_field;

        // TODO make separate function for visualization
        // // Step 5: Convert to output image
        // // Find max distance for normalization (excluding infinite values)
        // float max_dist = 0.0f;
        // for (float d : dist) {
        //     if (d != std::numeric_limits<float>::max()) {
        //         max_dist = std::max(max_dist, d);
        //     }
        // }
        // if (max_dist == 0.0f) max_dist = 1.0f; // Avoid divide by zero
    
        // distance_map_dest = Image(width, height);
        // for (int y = 0; y < height; ++y) {
        //     for (int x = 0; x < width; ++x) {
        //         float d = dist[y * width + x];
        //         uint8_t val;
                
        //         if (d == std::numeric_limits<float>::max()) {
        //             val = 255; // Max distance for unreachable pixels
        //         } else {
        //             val = static_cast<uint8_t>(std::min(255.0f, 255.0f * (d / max_dist)));
        //         }
                
        //         distance_map_dest.at(x, y) = Color(val, val, val);
        //     }
        // }
    }
    
    
    
    Image floatMapToGrayscaleImage(const std::vector<float>& data, Size size) {
        int width = size.width;
        int height = size.height;
        int count = width * height;
        if (data.size() != count) {
            throw std::runtime_error("floatMapToGrayscaleImage: size mismatch");
        }
    
        // Find min/max for normalization
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        for (float v : data) {
            if (!std::isfinite(v)) continue;
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
    
        if (min_val == max_val) max_val += 1.0f; // Avoid divide-by-zero
    
        Image img(width, height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float v = data[y * width + x];
                uint8_t gray = 255;
                if (std::isfinite(v)) {
                    float norm = (v - min_val) / (max_val - min_val);
                    gray = static_cast<uint8_t>(std::round(norm * 255.0f));
                }
                img.at(x, y) = Color(gray, gray, gray);
            }
        }
        return img;
    }
    
    



    void computeSobelGradients(const std::vector<float>& distance_map, Size size, 
        std::vector<float>& grad_x_dest, std::vector<float>& grad_y_dest) {

        int width = size.width;
        int height = size.height;

        grad_x_dest.assign(width * height, 0.0f);
        grad_y_dest.assign(width * height, 0.0f);

        auto at = [&](int x, int y) -> float {
            // Clamp edges
            x = std::clamp(x, 0, width - 1);
            y = std::clamp(y, 0, height - 1);
            return distance_map[y * width + x];
        };

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float gx =
                    -1.0f * at(x - 1, y - 1) + 1.0f * at(x + 1, y - 1) +
                    -2.0f * at(x - 1, y    ) + 2.0f * at(x + 1, y    ) +
                    -1.0f * at(x - 1, y + 1) + 1.0f * at(x + 1, y + 1);

                float gy =
                    -1.0f * at(x - 1, y - 1) + -2.0f * at(x, y - 1) + -1.0f * at(x + 1, y - 1) +
                    1.0f * at(x - 1, y + 1) +  2.0f * at(x, y + 1) +  1.0f * at(x + 1, y + 1);

                grad_x_dest[y * width + x] = gx;
                grad_y_dest[y * width + x] = gy;
            }
        }
    }








}