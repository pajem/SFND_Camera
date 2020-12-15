#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "structIO.hpp"

using namespace std;

void matchDescriptors(cv::Mat &imgSource, cv::Mat &imgRef, vector<cv::KeyPoint> &kPtsSource, vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      vector<cv::DMatch> &matches, string descriptorType, string matcherType, string selectorType)
{

    // configure matcher
    bool crossCheck = (selectorType == "SEL_KNN") ? false : true; // cross-check only works for K=1
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {

        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "FLANN matching";
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        double t = (double)cv::getTickCount();
        // k-nearest-neighbor matching
        int k = 2;
        std::vector<std::vector<cv::DMatch>> kMatches;
        matcher->knnMatch(descSource, descRef, kMatches, k);

        // filter matches using descriptor distance ratio test
        bool filtering = true;
        double descDistanceRatioThreshold = 0.8;
        for (auto& kMatch : kMatches) {
            if (!filtering) {
                matches.push_back(kMatch[0]);
            }
            else {
                double descDistanceRatio = kMatch[0].distance / kMatch[1].distance;
                if (descDistanceRatio < descDistanceRatioThreshold) {
                    matches.push_back(kMatch[0]);
                }
            }
        }

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }

    // visualize results
    cv::Mat matchImg = imgRef.clone();
    cv::drawMatches(imgSource, kPtsSource, imgRef, kPtsRef, matches,
                    matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    string windowName = "Matching keypoints between two camera images (best 50)";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, matchImg);
    cv::waitKey(0);
}

int main()
{
    cv::Mat imgSource = cv::imread("../images/img1gray.png");
    cv::Mat imgRef = cv::imread("../images/img2gray.png");

    vector<cv::KeyPoint> kptsSource, kptsRef; 
    readKeypoints("../dat/C35A5_KptsSource_BRISK_large.dat", kptsSource);
    readKeypoints("../dat/C35A5_KptsRef_BRISK_large.dat", kptsRef);

    cv::Mat descSource, descRef; 
    readDescriptors("../dat/C35A5_DescSource_BRISK_large.dat", descSource);
    readDescriptors("../dat/C35A5_DescRef_BRISK_large.dat", descRef);

    vector<cv::DMatch> matches;
    string matcherType = "MAT_FLANN"; 
    string descriptorType = "DES_BINARY"; 
    string selectorType = "SEL_KNN"; 
    matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType);
}