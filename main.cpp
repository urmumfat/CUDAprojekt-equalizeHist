#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void histogram(const unsigned char* in, int* histogram, int ile_pixeli) {
	std::fill(histogram, histogram + 256, 0);
	for (int i = 0; i < ile_pixeli; i++) {
		histogram[in[i]]++; 
	}
}

void dystrybuanta(const int* histogram, float* cdf) {
	cdf[0] = (float)histogram[0];
	for (int i = 1; i < 256; i++) {
		cdf[i] = cdf[i - 1] + (float)histogram[i];
	}
}

void LUT(const float* cdf, unsigned char* lut, float ile_pixeli) {
	float cdf_min = 0.0;
	for (int i = 0; i < 256; i++) {
		if (cdf[i] > cdf_min) {
			cdf_min = cdf[i];
			break;
		}
	}
	for (int i = 0; i < 256; i++) {
		lut[i] = cv::saturate_cast<uchar>(round((cdf[i] - cdf_min) / (ile_pixeli - cdf_min) * 255.0f));
	}
}

void stosuj_LUT(const unsigned char* in, unsigned char* out, const unsigned char* lut, int ile_pixeli) {
	for (int i = 0; i < ile_pixeli; i++) {
		out[i] = lut[in[i]];
	}
}

int main() {

	std::string path = "images.png";
	cv::Mat zdj = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (zdj.empty()) {
		std::cerr << "Nie mozna wczytac obrazu" << std::endl;
		return -1;
	}
	if (!zdj.isContinuous()) {
		zdj = zdj.clone();
	}
	int wysokosc = zdj.rows;
	int szerokosc = zdj.cols;
	int ile_pixeli = szerokosc * wysokosc;

	std::vector<unsigned char> zdj_in(zdj.data, zdj.data + ile_pixeli);
	std::vector<unsigned char> zdj_out(ile_pixeli);
	std::vector<int> hist(256);
	std::vector<float> dys(256);
	std::vector<unsigned char> lutable(256);

	auto start = std::chrono::high_resolution_clock::now();

	histogram(zdj_in.data(), hist.data(), ile_pixeli);
	dystrybuanta(hist.data(), dys.data());
	LUT(dys.data(), lutable.data(), ile_pixeli);
	stosuj_LUT(zdj_in.data(), zdj_out.data(), lutable.data(), ile_pixeli);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	std::cout << "czas wykonania: " << duration.count() << " ms" << std::endl;

	cv::Mat wynik (wysokosc, szerokosc, CV_8U, zdj_out.data());
	cv::Mat wzorzec;
	cv::equalizeHist(zdj, wzorzec);

	cv::imshow("oryginał", zdj);
	cv::imshow("Wynik equalizacji", wynik);
	cv::imshow("cv::equalizeHist", wzorzec);

	cv::waitKey(0);
	return 0;
}