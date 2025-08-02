#pragma once

#include "../../BaseTest.hpp"
#include "../../../Code/Utils/Image/Image.hpp"
#include "../../../Code/Modules/Mosaic/Mosaic.hpp"

#include <string>
#include <iostream>
#include <iomanip>

using namespace std;


namespace mosaic_gen::test { 

    class MosaicTest : public ::test::BaseTest {

        public: 

            explicit MosaicTest(Parameters& params): params_(params) {};
            
            void testConstructor();
            void testPipeline();
            void testSelectStroke();
            void testMask();
            void testRandomStart();
            void testFindThetaStroke();
            void testRingIntersections();
            void testMultipleRings();
            void testPlaceTileStroke();
            void testPlaceTileAllStrokes();

            void testSquareBorderPoints(); // TODO this belongs in geometry test as well
            
            void runAllTests() override;




        private: 


            bool verbose_;
            Parameters params_; 


           
    };

    





}