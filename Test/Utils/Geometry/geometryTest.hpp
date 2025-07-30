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


            bool testPCALength();

        
            
            void runAllTests() override;




        private: 



            string image_path_; 


           
    };

    





}