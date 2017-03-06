#include "ImageQualityAssessment.h"
#include <ctime>
int main(int argc,char** argv){
	string path = argv[1];
	string reference = path + "tem.bmp";
	string source = path + "org.bmp";
	Mat referenceImage = imread(reference, 0);
	Mat sourceImage = imread(source, 0);
	ImageQualityAssessment ssim = ImageQualityAssessment(referenceImage, sourceImage);
	clock_t ss = clock();
	cout << "ssim: "<<ssim.ssim() << endl;
	clock_t ee = clock();
	cout << "ssim cost " << (double)(ee - ss) / 1000 <<"s"<< endl;
	ss = clock();
	cout << "psnr: " << ssim.psnr() << endl;
	ee = clock();
	cout << "psnr cost " << (double)(ee - ss) / 1000 << "s" << endl;
	ssim.blockSize = 11;
	//imshow("ref", ssim.referenceImage);
	//imshow("src", ssim.sourceImage); 
	ss = clock();

	cout << "ssim: " << ssim.ssim(referenceImage, sourceImage) << endl;
	ee = clock();
	cout << "new ssim cost " << (double)(ee - ss) / 1000 << "s" << endl;

	cout << "hash: " << ssim.hashDistance() << endl;
	cout << "Dhash: " << ssim.DhashDistance() << endl;
	cout << "Phash: " << ssim.PhashDistance() << endl;
	//cout << ssim.sigma(referenceImage, 0, 0) << endl;
	waitKey();
	return 0;
}