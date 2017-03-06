#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define PSNR 1
#define CV_SSIM 2
#define MYSSIM 3
#define HASH 4
#define DHASH 5
#define PHASH 6

double blockIQA(Mat referenceImage, Mat sourceImage, int size, int IQAMethod = CV_SSIM);

double IQA(Mat referenceImage, Mat sourceImage, int IQAMethod = CV_SSIM);