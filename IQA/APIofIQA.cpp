#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "APIofIQA.h"

using namespace std;
using namespace cv;



//������ͼ��psnrֵ
double psnr(Mat referenceImage, Mat sourceImage){
	Mat diffImage;
	absdiff(referenceImage, sourceImage, diffImage);
	diffImage.convertTo(diffImage, CV_32F);
	Mat squareImage;
	multiply(diffImage, diffImage, squareImage);
	double sse = sum(squareImage)[0];
	if (sse <= 1e-6){
		return 0;
	}
	else{
		double mse = sse / referenceImage.total();
		double psnr = 10 * log10((255 * 255) / mse);
		return psnr;
	}
}

//������ͼ��ssimֵ(OPENCVʵ��)
double cvSsim(Mat referenceImage, Mat sourceImage){
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
	double ssim = mean(t3)[0];

	return ssim;
}

//������ͼ��ssimֵ����ʽʵ�֣�
double mySsim(Mat referenceImage, Mat sourceImage){
	const double C1 = 6.5025, C2 = 58.5225;		//C1=(K1*L)^2,C2=(K2*L)^2,C3=C2/2;K1=0.01,K2=0.03,L=255;
	double ssim = 0;
	int count = 1;
	int blockSize = 11;
	int nbBlockPerHeight = referenceImage.rows / blockSize;
	int nbBlockPerWidth = referenceImage.cols / blockSize;
	for (int k = 0; k < nbBlockPerHeight; k++)
	{
		for (int l = 0; l < nbBlockPerWidth; l++)
		{

			int m = k*blockSize;
			int n = l*blockSize;
			Mat imgx;
			referenceImage(Range(m, m + blockSize), Range(n, n + blockSize)).convertTo(imgx, CV_32F);
			Mat imgy;
			sourceImage(Range(m, m + blockSize), Range(n, n + blockSize)).convertTo(imgy, CV_32F);
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
			double t3 = t1 / t2;
			//cout << "t3: " << t3 << endl;
			ssim += t3;
			//ImageQualityAssessment localIQA = ImageQualityAssessment(imgx, imgy);
			//double localSsim = localIQA.ssim();	
			//ssim += localSsim;
		}
	}
	ssim /= nbBlockPerHeight*nbBlockPerWidth;
	return ssim;
}

Mat calcHashCode(Mat src){
	Mat image;
	resize(src, image, Size(8, 8));
	image /= 4;
	double average = mean(image)[0];
	Mat mask = (image >= average);
	return mask;
}

//��������ͼ����hash����
//����ͼ���ֵHashָ��
/*
�ŵ㣺�ٶȿ졣
ȱ�㣺�ܾ�ֵ��Ӱ��ǳ���
*/
double hashDistance(Mat referenceImage, Mat sourceImage){
	Mat referenceHashDistance = calcHashCode(referenceImage);
	Mat sourceHashDistance = calcHashCode(sourceImage);
	return countNonZero(referenceHashDistance != sourceHashDistance);
}


Mat calcPHashCode(Mat src){
	Mat image;
	resize(src, image, Size(32, 32));
	image.convertTo(image, CV_32FC1);
	dct(image, image);
	Rect roi(0, 0, 8, 8);
	double average = mean(image(roi))[0];
	Mat mask = (image(roi) >= average);
	return mask;
}
//��������ͼ����phash����
//����ͼ��pHashָ��
/*
�ŵ㣺���̱��Σ��ܹ�����٤��У����ֱ��ͼ���������Ӱ�졣
ȱ�㣺�ٶ�������
*/
double PhashDistance(Mat referenceImage, Mat sourceImage){
	Mat referencePHashDistance = calcPHashCode(referenceImage);
	Mat sourcePHashDistance = calcPHashCode(sourceImage);
	return countNonZero(referencePHashDistance != sourcePHashDistance);
}



Mat calcDHashCode(Mat src){
	Mat image;

	resize(src, image, Size(9, 8));
	Mat mask = Mat::zeros(Size(8, 8), CV_8UC1);
	for (int i = 0; i < image.cols - 1; i++)
	{
		for (int j = 0; j<image.rows; j++)
		{
			mask.at<uchar>(j, i) = (image.at<uchar>(j, i)>image.at<uchar>(j, i + 1));
		}
	}
	return mask;
}

//��������ͼ����dhash����
//����ͼ��dHashָ��
/*
�ŵ㣺�ٶȱ�pHash�죬Ч����hash�ã����ڽ���ʵ��
*/
double DhashDistance(Mat referenceImage, Mat sourceImage){
	Mat referenceDHashDistance = calcDHashCode(referenceImage);
	Mat sourceDHashDistance = calcDHashCode(sourceImage);
	return countNonZero(referenceDHashDistance != sourceDHashDistance);
}



double IQA(Mat referenceImage, Mat sourceImage, int IQAMethod){
	if (referenceImage.empty() || sourceImage.empty()){
		return -1;
	}
	if (referenceImage.size() != sourceImage.size()){
		return -2;
	}
	switch (IQAMethod)
	{
	case PSNR:
		return psnr(referenceImage, sourceImage);
	case CV_SSIM:
		return cvSsim(referenceImage, sourceImage);
	case MYSSIM:
		return mySsim(referenceImage, sourceImage);
	case HASH:
		return hashDistance(referenceImage, sourceImage);
	case DHASH:
		return DhashDistance(referenceImage, sourceImage);
	case PHASH:
		return PhashDistance(referenceImage, sourceImage);
	default:
		return -3;
	}
}

double blockIQA(Mat referenceImage, Mat sourceImage, int size,int IQAMethod){
	if (referenceImage.empty() || sourceImage.empty()){
		return -1;
	}
	if (referenceImage.size() != sourceImage.size()||size==0){
		return -2;
	}
	double ssim = 1;
	int unitHeight = referenceImage.rows / size;
	int unitWidth = referenceImage.cols / size;
	for (int i = 0; i < size;i++)
	{
		for (int j = 0; j < size;j++)
		{
			int m = i*unitHeight;
			int n = i*unitWidth;
			Mat imgx = referenceImage(Range(m, m + unitHeight), Range(n, n + unitWidth));
			Mat imgy = sourceImage(Range(m, m + unitHeight), Range(n, n + unitWidth));
			double blockScore = IQA(imgx, imgy, IQAMethod);
			if (blockScore < ssim){
				
				ssim = blockScore;
			}
		}
	}
	return ssim;
}

