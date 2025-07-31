#pragma once

#include "../../BaseTest.hpp"
#include "../../../Code/Utils/Image/Image.hpp"

#include <string>
#include <iostream>
#include <iomanip>

using namespace std;


namespace Graphics::test { 

    class GraphicsTest : public ::test::BaseTest {

        public: 

            explicit GraphicsTest(const string& image_path): image_path_(image_path) {};
            
            void testLine();
            void testLineMultiple();
            void testArrow();
            void testFillPolygon();
            void testDrawSquare();
            void testDrawSquareMultiple();
            void testRandom(); // I know this is not the random test library but for now

        
            
            void runAllTests() override;




        private: 


            bool verbose_;
            string image_path_; 


           
    };

    





}