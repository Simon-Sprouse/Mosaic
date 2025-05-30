cmake_minimum_required(VERSION 3.10)
project(MosaicTiler)

find_package(OpenCV REQUIRED)

add_executable(mosaic_tiler main.cpp image_process.cpp)
target_include_directories(mosaic_tiler PRIVATE ${OpenCv_INCLUDE_DIRS})
target_link_libraries(mosaic_tiler ${OpenCV_LIBS})
