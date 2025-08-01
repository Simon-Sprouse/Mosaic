#include "MosaicTest.hpp"

#include "../../../Code/Utils/Graphics/graphics.hpp"
#include "../../../Code/Utils/Io/io.hpp"
#include "../../../Code/Utils/Random/random.hpp"

namespace mosaic_gen::test { 

using image::Image;
using image::Size;
using image::Point;
using image::Color;




void MosaicTest::testConstructor() { 
    Mosaic mosaic(params_);
    // cout << "params_.tile_size: " << mosaic.params.tile_size << endl;
}


void MosaicTest::testPipeline() {

    Mosaic mosaic(params_);
    mosaic.contourPipeline();
    io::saveImage(mosaic.original, "../Results/original.jpg");
    io::saveImage(mosaic.resized, "../Results/resized.jpg");

    Image strokes_img(mosaic.resized.size());
    Graphics::drawStrokesRandomColor(strokes_img, mosaic.strokes);

    io::saveImage(strokes_img, "../Results/strokes.jpg");


}

void MosaicTest::testSelectStroke() { 

    // necessary progress
    Mosaic mosaic(params_);
    mosaic.contourPipeline();

    std::vector<int> stroke_ids = {0, 1, 2, 10};

    for (int stroke_id : stroke_ids) {
        mosaic.selectStroke(stroke_id);
        io::saveImage(mosaic.selected_stroke, "../Results/selected_stroke" + std::to_string(stroke_id) + ".jpg");
    }
    
}






void MosaicTest::runAllTests() { 
    cout << "RUNNING TESTS... " << endl;
    timeFunctionBar();

    chrono::duration<double> total_time(0.0);

    // call test functions
    total_time += timeFunction("Construct Mosaic", [&]() {testConstructor();});
    total_time += timeFunction("Contour Pipeline", [&]() {testPipeline();});
    total_time += timeFunction("Select Stroke", [&]() {testSelectStroke();});


   



    timeFunctionBar();
    printTotalTime(total_time);

    cout << endl;
}











}