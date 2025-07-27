#pragma once

#include "Image.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

namespace image::test { 

class ImageTest {

    public:

        ImageTest() : verbose_(true) {};
        explicit ImageTest(bool verbose) : verbose_(verbose) {}; 

        bool testStructPrint();
        bool testImageRuleOf5();
        bool testImagePrint();

        void runAllTests();




    private:

        static constexpr const char* RESET     = "\033[0m";
        static constexpr const char* BOLD      = "\033[1m";
        static constexpr const char* RED       = "\033[31m";
        static constexpr const char* GREEN     = "\033[32m";
        static constexpr const char* BOLD_RED      = "\033[1;31m";
        static constexpr const char* BOLD_GREEN    = "\033[1;32m";
        static constexpr const char* BOLD_YELLOW   = "\033[1;33m";

        bool verbose_;


        inline void printTestHeader(const string& test_name) { 
            printHorizontalBar();
            cout << BOLD << "Running Test: " << BOLD_YELLOW << test_name << RESET << endl;
        }

        inline void printTestFooter(bool test_passed) { 
            if (test_passed) { 
                cout << BOLD_GREEN << "... PASSED" << RESET << endl;
            }
            else { 
                cout << BOLD_RED << "... FAILED" << RESET << endl;
            }
        }

        inline void printHorizontalBar() { 
            cout << std::string(40, '~') << endl;
        }


        
        template <typename Func>
        void runFunction(const string& test_name, Func&& fn) {
            printTestHeader(test_name);
            bool test_passed = fn(); // run the test
            printTestFooter(test_passed);
        }


        template<typename T>
        bool checkEqual(const std::string& label, const T& actual, const T& expected) {
            if (actual == expected) {
                // only print success when verbose_
                if (verbose_) {
                    std::cout << label << BOLD_GREEN << ": [PASSED]\n" << RESET;
                }
                return true;
            }
            else {
                std::cout << label << BOLD_RED << ": [FAILED]\n" << RESET << RED
                        << "  Actual  : " << actual << "\n"
                        << "  Expected: " << expected << "\n" << RESET;
                return false;
            }
        }

        template<typename T>
        bool checkPrint(const std::string& label, const T& obj, const std::string& expected) {
            std::ostringstream oss;
            oss << obj;
            return checkEqual(label, oss.str(), expected);
        }




};

}




/*

List of things to emulate

🔲✅

✅ Color, Size, Point structs
🔲 Color, Size, Point structs correctness test
✅ Image class rule of 5
🔲 Image class rule of 5 correctness test




*/