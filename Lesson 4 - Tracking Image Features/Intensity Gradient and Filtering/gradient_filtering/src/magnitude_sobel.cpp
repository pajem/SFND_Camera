#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void magnitudeSobel()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1gray.png");

    // convert image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // apply smoothing using the GaussianBlur() function from the OpenCV
    cv::Mat gaussBlurImg;
    int kernelSize = 5;
    double stdDev = 2.0;
    cv::GaussianBlur(imgGray, gaussBlurImg, cv::Size(kernelSize, kernelSize), stdDev);

    // create filter kernels using the cv::Mat datatype both for x and y
    float sobelKernelX[9] = {
        -1, 0, +1,
        -2, 0, +2, 
        -1, 0, +1
    };
    cv::Mat sobelKernelXMat = cv::Mat(3, 3, CV_32F, sobelKernelX);
    float sobelKernelY[9] = {
        -1, -2, -1,
         0,  0,  0,
        +1, +2, +1
    };
    cv::Mat sobelKernelYMat = cv::Mat(3, 3, CV_32F, sobelKernelY);

    // apply filter using the OpenCv function filter2D()
    cv::Mat sobelXImg;
    cv::Mat sobelYImg;
    cv::filter2D(gaussBlurImg, sobelXImg, -1, sobelKernelXMat, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(gaussBlurImg, sobelYImg, -1, sobelKernelYMat, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // compute magnitude image based on the equation presented in the lesson
    cv::Mat magnitudeImg = imgGray.clone();
    for (int i = 0; i < magnitudeImg.rows; i++) {
      for (int j = 0; j < magnitudeImg.cols; j++) {
        uint8_t xVal = sobelXImg.at<uint8_t>(i, j);
        uint8_t yVal = sobelYImg.at<uint8_t>(i, j);
        magnitudeImg.at<uint8_t>(i, j) = std::sqrt(xVal * xVal + yVal * yVal);
      }
    }

    // show result
    string windowName = "Gaussian Blurring";
    cv::namedWindow(windowName, 1); // create window
    cv::imshow(windowName, magnitudeImg);
    cv::waitKey(0); // wait for keyboard input before continuing
}

int main()
{
    magnitudeSobel();
}