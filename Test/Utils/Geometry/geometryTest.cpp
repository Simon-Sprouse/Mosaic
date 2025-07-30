#include "geometryTest.hpp"
#include "../../../Code/Utils/Image/Image.hpp"
#include "../../../Code/Utils/Io/io.hpp"
#include "../../../Code/Utils/Random/random.hpp"


namespace Geometry::test { 

using image::Image;
using image::Color;
using image::Size;
using image::Point;






bool GeometryTest::testPCALength() { 

    bool all_passed;

    Image img = io::loadImage(image_path_);
    int num_points = 15;

    std::vector<Point> points = Random::samplePointsRandom(img, num_points);


    // TEST CV RESULTS
    cv::Mat data(points.size(), 2, CV_64F);
    for (size_t i = 0; i < points.size(); ++i) {
        data.at<double>(i, 0) = points[i].x;
        data.at<double>(i, 1) = points[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
    cv::Mat projected;
    pca.project(data, projected);
    double minVal, maxVal;
    cv::minMaxLoc(projected.col(0), &minVal, &maxVal);
    double answer = maxVal - minVal;


    cout << "CV PCA_length: " << answer << endl;











    all_passed &= checkEqual<double>("Pca length", answer, 420);



    return false;
}






void GeometryTest::runAllTests() { 



    runTruthTest("Pca Length", [&]() {return testPCALength();});

    




    cout << endl;




}


}