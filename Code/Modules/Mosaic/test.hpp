#pragma once

#include "Mosaic.hpp"

#include <chrono>
#include <string>


namespace mosaic_gen::Test { 

    class MosaicTest {

        public: 

            explicit MosaicTest(mosaic_gen::Mosaic& mosaic);

            void showDistanceVectorField();
            void squeebTest();
            void testFloodFillPoints();
            void testFloodFillFrontier();
            void testContours();
            void testSegmentSelection();
            void testSegmentOrder();
            void testIntersections();
            void visualizePlacementMethod();
            void visualizePlacementOrder();
            void visualizeFrontierOrder();
        
            
            void runAllTests();
            void runTimedProcess();



        private: 



            mosaic_gen::Mosaic& mosaic;

            void printTestHeader(const std::string& test_name);
            void printTestFooter(chrono::duration<double> elapsed);
            void printHorizontalBar();
            void printTotalTime(std::chrono::duration<double> total_time);

            template <typename Func>
            chrono::duration<double> timeFunction(const std::string& name, Func&& fn) {
                printTestHeader(name);
                auto start = std::chrono::high_resolution_clock::now();
                fn(); // run the test
                auto elapsed = std::chrono::high_resolution_clock::now() - start;
                printTestFooter(elapsed);
                return elapsed;
            }

    };

    





}