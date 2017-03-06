#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class ImageQualityAssessment
{
public:
	Mat referenceImage, sourceImage;
	int blockSize;
	ImageQualityAssessment(Mat image1, Mat image2);
	~ImageQualityAssessment();
	double ssim();
	double ssim(Mat& referenceImage, Mat& sourceImage, bool showProgress = false);

	double psnr();

	double hashDistance();
	double PhashDistance();
	double DhashDistance();
private:
	Mat calcHashCode(Mat src);
	Mat calcPHashCode(Mat src);
	Mat calcDHashCode(Mat src);
};

