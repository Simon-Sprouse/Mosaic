cmake_minimum_required(VERSION 3.10)
project(MosaicTiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(OpenCV REQUIRED)

# Collect all source files
add_executable(mosaic_tiler
    
    Modules/Mosaic/Mosaic.cpp

    Utils/Geometry/geometry.cpp
    Utils/Graphics/graphics.cpp
    Utils/Image/Image.cpp
    Utils/Image/io.cpp
    Utils/ImageProcess/ImageProcess.cpp
    Utils/Random/random.cpp

    ../Test/Modules/Mosaic/MosaicTest.cpp
    ../Test/Utils/Geometry/geometryTest.cpp
    ../Test/Utils/Graphics/graphicsTest.cpp
    ../Test/Utils/ImageProcess/ImageProcessTest.cpp


    main.cpp
)

# Add includes
target_include_directories(mosaic_tiler PRIVATE

    Modules/Mosiac

    Utils/Geometry
    Utils/Graphics
    Utils/Image
    Utils/ImageProcess
    Utils/Random

    ../Test


    
)

# Link OpenCV
target_link_libraries(mosaic_tiler ${OpenCV_LIBS})
