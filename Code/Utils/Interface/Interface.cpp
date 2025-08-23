#include "../Image/Image.hpp"
#include "../../Modules/Mosaic/Mosaic.hpp"
#include <stdint.h>
#include <iostream>
#include <string>

#include <emscripten/bind.h>
using namespace emscripten;

using image::Image;
using image::Point;
using image::Color;
using image::Size;

using mosaic_gen::Parameters;
using mosaic_gen::Mosaic;

std::string sayHello() {
    std::cout << "Hello from C++!" << std::endl;
    return "Hello from WASM!";
}

EMSCRIPTEN_BINDINGS(my_module) {


    emscripten::function("sayHello", &sayHello);



    // Bind Color struct
    value_object<Color>("Color")
        .field("r", &Color::r)
        .field("g", &Color::g)
        .field("b", &Color::b)
        .field("a", &Color::a)
        // Constructors: we can't bind multiple overloads directly,
        // but JS can construct with default and then set fields.
        ;

    // Bind Point struct
    value_object<Point>("Point")
        .field("x", &Point::x)
        .field("y", &Point::y)

        ;

    // Bind Size struct
    value_object<Size>("Size")
        .field("width", &Size::width)
        .field("height", &Size::height)
        ;

    class_<Parameters>("Parameters")
        .constructor<>()
        .property("resize_factor", &Parameters::resize_factor)
        .property("blur_kernel_size", &Parameters::blur_kernel_size)
        .property("blur_sigma", &Parameters::blur_sigma)
        .property("canny_threshold_1", &Parameters::canny_threshold_1)
        .property("canny_threshold_2", &Parameters::canny_threshold_2)
        .property("max_segment_angle_rad", &Parameters::max_segment_angle_rad)
        .property("min_segment_length", &Parameters::min_segment_length)
        .property("segment_angle_window", &Parameters::segment_angle_window)
        .property("tile_size", &Parameters::tile_size)
        .property("number_of_rings", &Parameters::number_of_rings)
        .property("initial_step", &Parameters::initial_step)
        .property("step_size", &Parameters::step_size)
        .property("min_intersection_distance", &Parameters::min_intersection_distance)
        .property("max_frontiers", &Parameters::max_frontiers)
        .property("flood_fill_neighbor_points", &Parameters::flood_fill_neighbor_points)
        .property("distance_from_center", &Parameters::distance_from_center)
        .property("random_background_points", &Parameters::random_background_points)
        .property("canny_resize_factor", &Parameters::canny_resize_factor)
        ;



    // Bind Image class
    class_<Image>("Image")
        // Constructors
        .constructor<>()
        .constructor<int,int>()
        .constructor<int,int,Color>()
        .constructor<Size>()
        .function("getWidth", &Image::getWidth)
        .function("getHeight", &Image::getHeight)
        .function("size", &Image::size)
        .function("empty", &Image::empty)
        .function("fill", &Image::fill)
        .function("clone", &Image::clone)
        .function("at", select_overload<Color&(int,int)>(&Image::at))
        .function("setPixel", &Image::setPixel)
        .function("getRawData", optional_override([](Image& self) -> uintptr_t {
                    uint8_t* data = self.rawData();
                    return reinterpret_cast<uintptr_t>(data);
                }))
        ;

    

    // Bind Mosaic class
    class_<Mosaic>("Mosaic")
        .constructor<const Parameters&>()
        .function("loadImageFromHeap", optional_override([](Mosaic& self, uintptr_t dataPtr, size_t size) {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(dataPtr);
            return self.loadImageFromBuffer(data, size);
        }))
        .function("stepK", &Mosaic::stepK)
        .function("renderImageRange", &Mosaic::renderImageRange)
        .function("renderDebugImageRange", &Mosaic::renderDebugImageRange)
        .function("setRenderPointer", &Mosaic::setRenderPointer)
        .function("getRenderPointer", &Mosaic::getRenderPointer)
        .function("resetCanvas", &Mosaic::resetCanvas)      
        .function("empty", &Mosaic::empty)
        .function("size", &Mosaic::size)
        .function("setParameters", &Mosaic::setParameters)
        .function("clearData", &Mosaic::clearData)
        .function("resizeOriginal", &Mosaic::resizeOriginal)
        .function("getContourImage", &Mosaic::getContourImage)
        .function("getCanvasPtr", &Mosaic::getCanvasPtr, allow_raw_pointer<Image*>())
        .function("getDebugCanvasPtr", &Mosaic::getDebugCanvasPtr, allow_raw_pointer<Image*>())
        .function("getOriginalImagePtr", &Mosaic::getOriginalImagePtr, allow_raw_pointer<Image*>())
        .function("getStrokesImagePtr", &Mosaic::getStrokesImagePtr, allow_raw_pointer<Image*>())
        .function("contourPipeline", &Mosaic::contourPipeline)
        .function("setDebugMode", &Mosaic::setDebugMode)
        ;
}
