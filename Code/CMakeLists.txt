cmake_minimum_required(VERSION 3.10)
project(MosaicTiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

# Collect all source files
add_executable(mosaic_tiler
    
    Modules/Mosaic/Mosaic.cpp
    Modules/Mosaic/test.cpp

    Utils/Geometry/geometry.cpp
    Utils/Graphics/graphics.cpp
    Utils/Image/Image.cpp
    Utils/Image/ImageTest.cpp
    Utils/Random/random.cpp

    main.cpp
)

# Add includes
target_include_directories(mosaic_tiler PRIVATE
    Modules/Mosaic
    Utils/Geometry
    Utils/Graphics
    Utils/Image
    Utils/Random
    Utils/IO
)

# Link OpenCV
target_link_libraries(mosaic_tiler ${OpenCV_LIBS})
