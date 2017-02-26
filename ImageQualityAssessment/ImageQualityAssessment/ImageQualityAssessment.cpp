#include "ImageQualityAssessment.h"


ImageQualityAssessment::ImageQualityAssessment(Mat image1, Mat image2) :referenceImage(image1), sourceImage(image2)
{
}


ImageQualityAssessment::~ImageQualityAssessment()
{
}

double ImageQualityAssessment::ssim(){
	const double C1 = 6.5025, C2 = 58.5225;		//C1=(K1*L)^2,C2=(K2*L)^2,C3=C2/2;K1=0.01,K2=0.03,L=255;
	Mat Ix, Iy;
	referenceImage.convertTo(Ix, CV_32F);
	sourceImage.convertTo(Iy, CV_32F);
	Mat Ixx = Ix.mul(Ix);
	Mat Iyy = Iy.mul(Iy);
	Mat Ixy = Ix.mul(Iy);
	Mat mux, muy;
	GaussianBlur(Ix, mux, Size(11, 11), 1.5);
	GaussianBlur(Iy, muy, Size(11, 11), 1.5);
	Mat muxx = mux.mul(mux);
	Mat muyy = muy.mul(muy);
	Mat muxy = mux.mul(muy);
	Mat sigmalxx, sigmalyy, sigmalxy;
	GaussianBlur(Ixx, sigmalxx, Size(11, 11), 1.5);
	sigmalxx -= muxx;
	GaussianBlur(Iyy, sigmalyy, Size(11, 11), 1.5);
	sigmalyy -= muyy;
	GaussianBlur(Ixy, sigmalxy, Size(11, 11), 1.5);
	sigmalxy -= muxy;
	Mat t1 = (2 * muxy + C1).mul(2 * sigmalxy + C2);
	Mat t2 = (muxx + muyy + C1).mul(sigmalxx + sigmalyy + C2);
	Mat t3;
	divide(t1, t2, t3);
	Scalar mssim = mean(t3);
	cout << "mssim: " << mssim << endl;
	double ssim = (mssim.val[0]+ mssim.val[1]+ mssim.val[2]) / 3;
	return ssim;
}


double ImageQualityAssessment::ssim(Mat& referenceImage, Mat& sourceImage, bool showProgress){
	const double C1 = 6.5025, C2 = 58.5225;		//C1=(K1*L)^2,C2=(K2*L)^2,C3=C2/2;K1=0.01,K2=0.03,L=255;
	double ssim = 0;
	
	int nbBlockPerHeight = referenceImage.rows / blockSize;
	int nbBlockPerWidth = referenceImage.cols / blockSize;
	for (int k = 0; k < nbBlockPerHeight ; k++)
	{
		for (int l = 0; l < nbBlockPerWidth ; l++)
		{
			
			int m = k*blockSize;
			int n = l*blockSize;
			Mat imgx;
			referenceImage(Range(m, m + blockSize), Range(n, n + blockSize)).convertTo(imgx,CV_32F);
			Mat imgy;
			sourceImage(Range(m, m + blockSize), Range(n, n + blockSize)).convertTo(imgy,CV_32F);
			Mat meanx, sdx, meany, sdy;
			meanStdDev(imgx, meanx, sdx);
			meanStdDev(imgy, meany, sdy);
			double avgx = meanx.at<double>(0, 0);
			double avgy = meany.at<double>(0, 0);
			double sigmax = sdx.at<double>(0, 0);
			double sigmay = sdy.at<double>(0, 0);
			Mat imgxy = Mat::zeros(blockSize, blockSize, imgx.depth());
			
			multiply(imgx, imgy, imgxy);
			double sigmaxy = mean(imgxy)[0] - avgx*avgy;
			double t1 = (2 * avgx*avgy + C1)*(2 * sigmaxy + C2);
			double t2 = (avgx*avgx + avgy*avgy + C1)*(sigmax*sigmax + sigmay*sigmay + C2);
			ssim += t1 / t2;
		}
		if (showProgress){
			cout << "\r>>SSIM [" << (int)((((double)k) / nbBlockPerHeight) * 100) << "%]" << ssim << endl;;
		}
		ssim /= nbBlockPerHeight*nbBlockPerWidth;
	}
	if (showProgress){
		cout << "\r>>SSIM[100%]" << endl;
		cout << "SSIM: " << ssim << endl;
	}
	return ssim;
}