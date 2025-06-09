cmake_minimum_required(VERSION 3.10)
project(MosaicTiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(mosaic_tiler 
    main.cpp 
    mosaic.cpp
    graphics.cpp 
    image_process.cpp)
target_include_directories(mosaic_tiler PRIVATE ${OpenCv_INCLUDE_DIRS})
target_link_libraries(mosaic_tiler ${OpenCV_LIBS})
