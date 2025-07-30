#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace std;

namespace test { 

class BaseTest { 

    public: 

        virtual ~BaseTest() = default;

        virtual void runAllTests() {};

        void setVerbose(bool isVerbose) {verbose_ = isVerbose;};


    protected:

        static constexpr const char* RESET     = "\033[0m";
        static constexpr const char* BOLD      = "\033[1m";
        static constexpr const char* RED       = "\033[31m";
        static constexpr const char* GREEN     = "\033[32m";
        static constexpr const char* BOLD_RED      = "\033[1;31m";
        static constexpr const char* BOLD_GREEN    = "\033[1;32m";
        static constexpr const char* BOLD_YELLOW   = "\033[1;33m";

        bool verbose_ = false;


        /*
            For timed functions (especially cases with no truth value)
        */

        inline void timeFunctionHeader(const std::string& test_name) {
            std::cout << std::left << std::setw(40) << ("[Running] " + test_name)
                        << " | ";
            std::cout.flush(); // flush in case timing starts immediately after
        }
        
        inline void timeFunctionFooter(std::chrono::duration<double> elapsed) {
            std::cout << std::right << std::setw(12)
                        << std::fixed << std::setprecision(4)
                        << elapsed.count() << " s" << std::endl;
        }
        
        inline void timeFunctionBar() { 
            cout << std::string(41, '~') << " " << std::string(15, '~') << endl;
        }
        
        inline void printTotalTime(std::chrono::duration<double> total_time) { 
        
            cout << std::left << std::setw(40) << "[Total Time]"
            << std::right << std::setw(15) << std::fixed << std::setprecision(4)
            << total_time.count() << " s" << endl;
        }

        template <typename Func>
        chrono::duration<double> timeFunction(const std::string& name, Func&& fn) {
            timeFunctionHeader(name);
            auto start = std::chrono::high_resolution_clock::now();
            fn(); // run the test
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            timeFunctionFooter(elapsed);
            return elapsed;
        }



        /*
            For correctness functions
        */

        inline void truthTestHeader(const string& test_name) { 
            truthTestBar();
            cout << BOLD << "Running Test: " << BOLD_YELLOW << test_name << RESET << endl;
        }

        inline void truthTestFooter(bool test_passed) { 
            if (test_passed) { 
                cout << BOLD_GREEN << "... PASSED" << RESET << endl;
            }
            else { 
                cout << BOLD_RED << "... FAILED" << RESET << endl;
            }
        }

        inline void truthTestBar() { 
            cout << std::string(40, '~') << endl;
        }


        
        template <typename Func>
        void runTruthTest(const string& test_name, Func&& fn) {
            truthTestHeader(test_name);
            bool test_passed = fn(); // run the test
            truthTestFooter(test_passed);
        }


        template<typename T>
        bool checkEqual(const std::string& label, const T& guess, const T& gt) {
            if (guess == gt) {
                // only print success when verbose_
                if (verbose_) {
                    std::cout << label << BOLD_GREEN << ": [PASSED]\n" << RESET;
                }
                return true;
            }
            else {
                std::cout << label << BOLD_RED << ": [FAILED]\n" << RESET << RED
                        << "  Failed Attempt  : " << guess << "\n"
                        << "  Correct Answer  : " << gt << "\n" << RESET;
                return false;
            }
        }


        template<typename T>
        bool checkWithTolerance(const std::string& label, const T& guess, const T& gt, double ratio_tol = 0.05) {
            if (gt == 0) {
                // Avoid division by zero; fall back to absolute check
                bool passed = std::abs(guess) <= ratio_tol;
                if (passed && verbose_) {
                    std::cout << label << BOLD_GREEN << ": [PASSED]\n" << RESET;
                } else if (!passed) {
                    std::cout << label << BOLD_RED << ": [FAILED]\n" << RESET << RED
                            << "  Failed Attempt  : " << guess << "\n"
                            << "  Correct Answer  : " << gt << "\n"
                            << "  Ratio tolerance : " << ratio_tol << "\n"
                            << RESET;
                }
                return passed;
            }

            double ratio_error = std::abs((guess / gt) - 1.0);
            bool passed = ratio_error <= ratio_tol;

            if (passed) {
                if (verbose_) {
                    std::cout << label << BOLD_GREEN << ": [PASSED]\n" << RESET;
                }
            } else {
                std::cout << label << BOLD_RED << ": [FAILED]\n" << RESET << RED
                        << "  Failed Attempt  : " << guess << "\n"
                        << "  Correct Answer  : " << gt << "\n"
                        << "  Ratio Error     : " << ratio_error << "\n"
                        << "  Allowed Tol     : " << ratio_tol << "\n"
                        << RESET;
            }

            return passed;
        }







        template<typename T>
        bool checkPrint(const std::string& label, const T& obj, const std::string& expected) {
            std::ostringstream oss;
            oss << obj;
            return checkEqual(label, oss.str(), expected);
        }





};




}