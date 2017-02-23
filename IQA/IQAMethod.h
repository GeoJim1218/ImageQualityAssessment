#pragma once
#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class IQAMethod
{
public:
	Mat referenceImage, sourceImage;
	double ssim();
	IQAMethod();
	~IQAMethod();
};

