#pragma once

#include "../../BaseTest.hpp"

#include <string>
#include <iostream>
#include <iomanip>

using namespace std;


namespace Geometry::test { 

    class GeometryTest : public ::test::BaseTest {

        public: 

            explicit GeometryTest(const string& image_path): image_path_(image_path) {};
            // TODO dynamically set verbosity
            // GeometryTest(const string& image_path, bool verbose) : image_path_(image_path), verbose_(verbose) {};

            bool testComputeMean();
            bool testCovarianceMatrix();
            bool testFirstEigenVector();
            bool testPcaLength();
            bool testPcaDirection();

        
            
            void runAllTests() override;




        private: 


            bool verbose_;
            string image_path_; 


           
    };

    





}