#include "../Image/Image.hpp"
#include "../Random/random.hpp"
#include "../../Modules/Mosaic/Mosaic.hpp"
#include <cstdint>
#include <iostream>


extern "C" {

    const char* helloWorld() {
    
        return "Hello from Interface.cpp";
    }

    


    image::Image* loadImageFromBytes(uint8_t* data, size_t length) {
        return new image::Image(image::fromEncodedBuffer(data, length));
    }

    const char* getSizeStr(image::Image* img) { 
        static char buffer[32]; // big enough for "12345,12345\0"
        std::string result = img->size().toString();
        strncpy(buffer, result.c_str(), sizeof(buffer));
        buffer[sizeof(buffer) - 1] = '\0'; // null-terminate
        return buffer;
    }
    

    bool empty(image::Image* img) { 
        return img->empty();
    }

    uint8_t* getData(image::Image* img) {
        return img->rawData();
    }
    

    void deleteImage(image::Image* img) { 
        delete img;
    }



    



    mosaic_gen::Mosaic* loadMosaicFromBytes(uint8_t* data, size_t length) { 

        std::cout << "loading Mosaic from bytes" << std::endl;



        mosaic_gen::Parameters params;
        params.resize_factor = 1.5;
        params.blur_kernel_size = 3;
        params.blur_sigma = 1.4;
        params.canny_threshold_1 = 50;
        params.canny_threshold_2 = 100;
        params.max_segment_angle_rad = 100 * M_PI / 180.0; // TODO why is this in rad
        params.min_segment_length = 20;
        params.segment_angle_window = 10;
        params.tile_size = 10;
        params.number_of_rings = 5;
        params.step_size = 1 * params.tile_size;
        params.min_intersection_distance = 1.5 * params.tile_size;
        params.max_frontiers = 20;
        params.flood_fill_neighbor_points = 4;
        params.distance_from_center = 1.5 * params.tile_size;
        params.random_background_points = 50000;
        params.tiles_per_frame = 20;
        params.jitter_map.insert({4, 0});
        params.jitter_map.insert({8, 1});
        params.jitter_map.insert({12, 10});






        mosaic_gen::Mosaic* mosaic_ptr = new mosaic_gen::Mosaic(params);
        mosaic_ptr->loadImageFromBuffer(data, length);

        std::cout << "mosaic->size(): " << mosaic_ptr->size() << std::endl;

        return mosaic_ptr;
    }


    uint8_t* getMosaicOutput(mosaic_gen::Mosaic* mosaic_ptr) { 
        mosaic_ptr->runAll();

        std::cout << "mosaic_ptr->getCanvas(): " << mosaic_ptr-> getCanvas() << std::endl;

        return mosaic_ptr->getCanvas().rawData();
    }





    const char* getMosaicSizeStr(mosaic_gen::Mosaic* mosaic_ptr) {
        std::string result = mosaic_ptr->size().toString();
        char* buffer = (char*)malloc(result.size() + 1);  // +1 for null terminator
        std::strcpy(buffer, result.c_str());
        return buffer;
    }
    

    bool mosaicIsEmpty(mosaic_gen::Mosaic* mosaic) { 
        return mosaic->empty();
    }

    void deleteMosaic(mosaic_gen::Mosaic* mosaic) { 
        delete mosaic;
    }













   
}
