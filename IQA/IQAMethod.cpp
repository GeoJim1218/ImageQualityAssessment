#include "IQAMethod.h"


IQAMethod::IQAMethod()
{
}


IQAMethod::~IQAMethod()
{
}

double IQAMethod::ssim(){
	if (referenceImage.empty()||sourceImage.empty())
	{
		return 0;
	}
	const double C1=0, C2=0;
	Mat Ix, Iy;
	referenceImage.convertTo(Ix, CV_32F);
	sourceImage.convertTo(Iy, CV_32F);
	Mat Ixx = Ix.mul(Ix);
	Mat Iyy = Iy.mul(Iy);
	Mat Ixy = Ix.mul(Iy);

	Mat Mux, Muy;
	GaussianBlur(Ix, Mux, Size(11, 11), 1.5);
	GaussianBlur(Iy, Muy, Size(11, 11), 1.5);
}
