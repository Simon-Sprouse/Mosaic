// #pragma once

// #include <string>
// #include <iostream>
// #include <iomanip>

// using namespace std;


// namespace image::process::test { 

//     class ProcessTest {

//         public: 

//             explicit ProcessTest(const string& image_path): image_path_(image_path) {};

//             void loadImage();
//             void squeebTest();
//             void testResize();
//             void testGrayscale();
//             void testBlur();
//             void testSobel();
//             void testCanny();
//             void testContours();

        
            
//             void runAllTests();




//         private: 



//             string image_path_; 


//             inline void printTestHeader(const std::string& test_name) {
//                 std::cout << std::left << std::setw(40) << ("[Running] " + test_name)
//                             << " | ";
//                 std::cout.flush(); // flush in case timing starts immediately after
//             }
            
//             inline void printTestFooter(std::chrono::duration<double> elapsed) {
//                 std::cout << std::right << std::setw(12)
//                             << std::fixed << std::setprecision(4)
//                             << elapsed.count() << " s" << std::endl;
//             }
            
//             inline void printHorizontalBar() { 
//                 cout << std::string(41, '~') << " " << std::string(15, '~') << endl;
//             }
            
//             inline void printTotalTime(std::chrono::duration<double> total_time) { 
            
//                 cout << std::left << std::setw(40) << "[Total Time]"
//                 << std::right << std::setw(15) << std::fixed << std::setprecision(4)
//                 << total_time.count() << " s" << endl;
//             }

//             template <typename Func>
//             chrono::duration<double> timeFunction(const std::string& name, Func&& fn) {
//                 printTestHeader(name);
//                 auto start = std::chrono::high_resolution_clock::now();
//                 fn(); // run the test
//                 auto elapsed = std::chrono::high_resolution_clock::now() - start;
//                 printTestFooter(elapsed);
//                 return elapsed;
//             }

//     };

    





// }