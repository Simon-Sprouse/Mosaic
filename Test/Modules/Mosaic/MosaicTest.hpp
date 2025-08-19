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

            explicit MosaicTest(Parameters& params, string image_path, string save_dir) 
                : params_(params), image_path_(image_path), save_dir_(save_dir) {};
            
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
            void testVectorField();
            void testFloodFill();
            void testGapFill();

            void testReconstructImage();


            void testStepOnce();
            void testStepK();

            void testCanny();
            
            void runAllTests() override;




        private: 


            bool verbose_;
            Parameters params_; 
            string image_path_;
            string save_dir_;


           
    };

    





}