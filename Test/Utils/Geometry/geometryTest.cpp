#include <opencv2/opencv.hpp>

#include "geometryTest.hpp"

#include "../../../Code/Utils/Geometry/geometry.hpp"
#include "../../../Code/Utils/Image/Image.hpp"
#include "../../../Code/Utils/Image/io.hpp"
#include "../../../Code/Utils/Random/random.hpp"


namespace Geometry::test { 

using image::Image;
using image::Color;
using image::Size;
using image::Point;
using image::io::loadImageFileSystem;





bool GeometryTest::testComputeMean() { 
    bool all_passed = true;

    Image img = loadImageFileSystem(image_path_);
    int num_points = 15;
    std::vector<Point> points = Random::randomPointsVector(img.size(), num_points);

    // Convert to a matrix of 2D vectors
    cv::Mat data(points.size(), 1, CV_64FC2);
    for (size_t i = 0; i < points.size(); ++i) {
        data.at<cv::Vec2d>(i, 0) = cv::Vec2d(points[i].x, points[i].y);
    }

    // Compute mean
    cv::Scalar mean_cv = cv::mean(data);

    Point answer;
    answer.x = mean_cv[0];
    answer.y = mean_cv[1];

    Point attempt = Geometry::computeMean(points);

    all_passed &= checkEqual<Point>("Compute Mean", attempt, answer);

    return all_passed;
}


bool GeometryTest::testCovarianceMatrix() { 
    bool all_passed = true;

    Image img = loadImageFileSystem(image_path_);
    int num_points = 15;
    std::vector<Point> points = Random::randomPointsVector(img.size(), num_points);

    // use gt library to build answer

    // Prepare data for OpenCV
    cv::Mat data(points.size(), 2, CV_64F);
    for (size_t i = 0; i < points.size(); ++i) {
        data.at<double>(i, 0) = points[i].x;
        data.at<double>(i, 1) = points[i].y;
    }
    // Compute covariance matrix with OpenCV
    cv::Mat mean_cv;
    cv::Mat covMat;
    cv::calcCovarMatrix(data, covMat, mean_cv, cv::COVAR_NORMAL | cv::COVAR_ROWS, CV_64F);
    covMat = covMat / (points.size());  // normalize by N, matching your code's normalization
    // Build answer as std::array<std::array<double, 2>, 2>
    std::array<std::array<double, 2>, 2> answer {{
        { covMat.at<double>(0, 0), covMat.at<double>(0, 1) },
        { covMat.at<double>(1, 0), covMat.at<double>(1, 1) }
    }};
   

    Point mean = Geometry::computeMean(points);
    auto centered_points = Geometry::centerData(points, mean); // function expects centered points
    std::array<std::array<double, 2>, 2> attempt = Geometry::computeCovarianceMatrix(centered_points);

    // unpack arrays to check
    double ratio_tolerance = 0.01;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            all_passed &= checkWithTolerance<double>(
                "Covariance Matrix[" + std::to_string(i) + "][" + std::to_string(j) + "]",
                attempt[i][j],
                answer[i][j],
                ratio_tolerance
            );
            // cout << "attempt[" << i << "][" << "j" << "]: " << attempt[i][j] << endl;
            // cout << "answer[" << i << "][" << "j" << "]: " << answer[i][j] << endl;
        }
    }

    return all_passed;
}



bool GeometryTest::testFirstEigenVector() { 
    bool all_passed = true;

    Image img = loadImageFileSystem(image_path_);
    int num_points = 15;
    std::vector<Point> points = Random::randomPointsVector(img.size(), num_points);
    Point mean = Geometry::computeMean(points);
    auto centered_points = Geometry::centerData(points, mean); // function expects centered points
    auto cov = Geometry::computeCovarianceMatrix(centered_points);
    Vec2d attempt = Geometry::computeFirstEigenvector(cov);

    // use gt library to build answer

    // Use OpenCV to compute eigenvalues and eigenvectors
    cv::Mat covMat(2, 2, CV_64F);
    covMat.at<double>(0, 0) = cov[0][0];
    covMat.at<double>(0, 1) = cov[0][1];
    covMat.at<double>(1, 0) = cov[1][0];
    covMat.at<double>(1, 1) = cov[1][1];

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covMat, eigenvalues, eigenvectors);

    // First row of eigenvectors is the dominant eigenvector
    Vec2d answer;
    answer.x = eigenvectors.at<double>(0, 0);
    answer.y = eigenvectors.at<double>(0, 1);

    // Ensure consistent direction (flip sign if needed)
    if ((attempt.x * answer.x + attempt.y * answer.y) < 0) {
        answer.x *= -1;
        answer.y *= -1;
    }
   
    

    double ratio_tolerance = 0.01;
    all_passed &= checkWithTolerance<double>("Eigen Vector x", attempt.x, answer.x, ratio_tolerance);
    all_passed &= checkWithTolerance<double>("Eigen Vector y", attempt.y, answer.y, ratio_tolerance);
    // cout << "attempt: " << attempt << endl;
    // cout << "answer: " << answer << endl;

    return all_passed;
}










bool GeometryTest::testPcaLength() { 

    bool all_passed = true;

    Image img = loadImageFileSystem(image_path_);
    int num_points = 15;

    std::vector<Point> points = Random::randomPointsVector(img.size(), num_points);
    double attempt = Geometry::pcaLength(points);

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

    

    double ratio_tolerance = 0.01;
    all_passed &= checkWithTolerance<double>("Pca length", attempt, answer, ratio_tolerance);



    return all_passed;
}


bool GeometryTest::testPcaDirection() { 

    bool all_passed = true;

    Image img = loadImageFileSystem(image_path_);
    int num_points = 15;

    std::vector<Point> points = Random::randomPointsVector(img.size(), num_points);
    Vec2d attempt = Geometry::pcaDirection(points);


    // Use library for GT
    // Build matrix for PCA
    cv::Mat data(points.size(), 2, CV_64F);
    for (size_t i = 0; i < points.size(); ++i) {
        data.at<double>(i, 0) = points[i].x;
        data.at<double>(i, 1) = points[i].y;
    }

    // Run PCA to get dominant direction
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
    cv::Vec2d direction = pca.eigenvectors.row(0);
    Vec2d answer(direction[0], direction[1]);


    double ratio_tolerance = 0.01;
    all_passed &= checkWithTolerance<double>("Eigen Vector x", attempt.x, answer.x, ratio_tolerance);
    all_passed &= checkWithTolerance<double>("Eigen Vector y", attempt.y, answer.y, ratio_tolerance);

    return all_passed;
}











void GeometryTest::runAllTests() { 


    runTruthTest("Compute Mean", [&]() {return testComputeMean();});
    runTruthTest("Covariance Matrix", [&]() {return testCovarianceMatrix();});
    runTruthTest("First Eigenvector", [&]() {return testFirstEigenVector();});
    runTruthTest("Pca Length", [&]() {return testPcaLength();});
    runTruthTest("Pca Direction", [&]() {return testPcaDirection();});

    




    cout << endl;




}


}