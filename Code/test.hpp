#pragma once

#include "Mosaic.hpp"


#include <string>


namespace Test { 

    void showDistanceVectorField(mosaic_gen::Mosaic& mosaic);
    void squeebTest();
    void testFloodFillPoints(mosaic_gen::Mosaic& mosaic);

    void printTestHeader(const std::string& test_name);
    void printTestFooter(chrono::duration<double> elapsed);
    void runAllTests(mosaic_gen::Mosaic& mosaic);
    void runTimedProcess(mosaic_gen::Mosaic& mosaic);
    void oldMain(mosaic_gen::Mosaic& mosaic);




}