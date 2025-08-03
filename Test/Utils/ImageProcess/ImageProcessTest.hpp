#pragma once

#include "../../BaseTest.hpp"
#include "../../../Code/Utils/Image/Image.hpp"

namespace image::process::test { 


    class ProcessTest : public ::test::BaseTest { 


        public: 

            explicit ProcessTest(const string& image_path): image_path_(image_path) {};

            void testLoadImage();
//             void squeebTest();
//             void testResize();
//             void testGrayscale();
//             void testBlur();
//             void testSobel();
//             void testCanny();
//             void testContours();

            void testComputeDistanceField(); // TODO needs visualization, maybe image constructor from float vector
            void testDistanceGradients();
            void testVectorField();

            

        
            
            void runAllTests() override;




        private: 



            string image_path_; 












    };






}