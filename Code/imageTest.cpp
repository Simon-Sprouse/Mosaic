#include "imageTest.hpp"

#include <iostream>

using namespace std;


namespace image_test { 


void testColor() { 
    
    image::Color my_color(255);

    cout << "Testing color: " << my_color << endl;

}

void testPoint() { 

    image::Point my_point(420, 69);

    cout << "Testing point: " << my_point << endl;




}






}