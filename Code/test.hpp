#pragma once

#include "Mosaic.hpp"


#include <string>


namespace Test { 

    void showDistanceVectorField(mosaic_gen::Mosaic& mosaic);
    void squeebTest();
    void testFloodFillPoints(mosaic_gen::Mosaic& mosaic);
    void testFloodFillFrontier(mosaic_gen::Mosaic& mosaic);
    void testContours(mosaic_gen::Mosaic& mosaic);
    void testSegmentSelection(mosaic_gen::Mosaic& mosaic);
    void testSegmentOrder(mosaic_gen::Mosaic& mosaic);
    void testIntersections(mosaic_gen::Mosaic& mosaic);
    void visualizePlacementMethod(mosaic_gen::Mosaic& mosaic);
    void visualizePlacementOrder(mosaic_gen::Mosaic& mosaic);
    void visualizeFrontierOrder(mosaic_gen::Mosaic& mosaic);

    void printTestHeader(const std::string& test_name);
    void printTestFooter(chrono::duration<double> elapsed);
    void runAllTests(mosaic_gen::Mosaic& mosaic);
    void runTimedProcess(mosaic_gen::Mosaic& mosaic);





}