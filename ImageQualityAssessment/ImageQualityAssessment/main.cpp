#include "ImageQualityAssessment.h"

int main(int argc,char** argv){
	string path = argv[1];
	string reference = path + "tem.bmp";
	string source = path + "org.bmp";
	Mat referenceImage = imread(reference, 0);
	Mat sourceImage = imread(source, 0);
	ImageQualityAssessment ssim = ImageQualityAssessment(referenceImage, sourceImage);
	cout << "ssim: "<<ssim.ssim() << endl;
	ssim.blockSize = 11;
	//imshow("ref", ssim.referenceImage);
	//imshow("src", ssim.sourceImage);
	cout << "ssim: " << ssim.ssim(referenceImage, sourceImage,true) << endl;
	//cout << ssim.sigma(referenceImage, 0, 0) << endl;
	waitKey();
	return 0;
}