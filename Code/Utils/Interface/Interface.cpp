#include "../Image/Image.hpp"
#include "../../Modules/Mosaic/Mosaic.hpp"
#include <stdint.h>
#include <iostream>

#include <emscripten/bind.h>
using namespace emscripten;

using image::Image;
using image::Point;
using image::Color;
using image::Size;

using mosaic_gen::Parameters;
using mosaic_gen::Mosaic;

EMSCRIPTEN_BINDINGS(my_module) {


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
        .property("tiles_per_frame", &Parameters::tiles_per_frame);


    // Bind Image class
    class_<Image>("Image")
        // Constructors
        .constructor<>()
        .constructor<int,int>()
        .constructor<int,int,Color>()
        .constructor<Size>()
        // .constructor<Size,Color>() // constructors only overloaded with parameter count?? 
        //.constructor<Size,std::vector<float>>() // Vector binding more complex; omit or add if needed
        .function("getWidth", &Image::getWidth)
        .function("getHeight", &Image::getHeight)
        .function("size", &Image::size)
        .function("empty", &Image::empty)
        .function("fill", &Image::fill)
        .function("clone", &Image::clone)
        .function("at", select_overload<Color&(int,int)>(&Image::at))
        .function("setPixel", &Image::setPixel)
        .function("rawData", select_overload<const uint8_t*() const>(&Image::rawData), allow_raw_pointer<const uint8_t*>())
        ;

    

    // Bind Mosaic class
    class_<Mosaic>("Mosaic")
        .constructor<const Parameters&>()
        .function("loadImageFromHeap", optional_override([](Mosaic& self, uintptr_t dataPtr, size_t size) {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(dataPtr);
            return self.loadImageFromBuffer(data, size);
        }))
        .function("runAll", &Mosaic::runAll)
        .function("getRawData", optional_override([](Mosaic& self) -> uintptr_t {
                    uint8_t* data = self.getRawData();
                    return reinterpret_cast<uintptr_t>(data);
                }))        
        .function("empty", &Mosaic::empty)
        .function("size", &Mosaic::size)
        // You can add more Mosaic methods as needed
        ;
}
