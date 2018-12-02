//HDF5 lib
#include <hdf5/serial/H5Cpp.h>
#include <hdf5/serial/hdf5.h>

//Default lib
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <byteswap.h>
#include <bitset>
#include <stdlib.h>

//OPENCV lib
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace H5;

#define N 60000
#define ROWS 28
#define COLS 28
#define PIXEL 784
#define N2 50000

void read(const int n, const char* v[]) {
	if (n < 3) {
	  cerr << "NAME and LOCATION" << endl;
	  return;
	}

	const char* filename = v[1];

	cout << filename << " opening" << endl;

	H5File file(filename, H5F_ACC_RDWR);

	cout << filename << " opened" << endl;

	DataSet dataset = file.openDataSet("data");

	float* value = new float[N * PIXEL];

	dataset.read(value, PredType::NATIVE_FLOAT);

	cout.precision(1);

	for (int i = 0; i != 27; ++i) {
		for (int j = 0; j != 27; ++j) {
			cout << value[j + (i * 28)] << " ,";
		}
		cout << endl;
	}


}

void write(const int n, const char* v[]) {
	if (n < 3) {
	  cerr << "name img_loc label_loc" << endl;
	  return;
	}

	const char* filename = v[1];
	const char* img_loc = v[2];
	const char* label_loc = v[3];

	H5File file(filename, H5F_ACC_TRUNC);

	hsize_t dim[4]{60000, 1, ROWS, COLS};

	hsize_t dim2[4]{60000, 1, 1, 1};

	DataSpace dataspace(4, dim);

	DataSpace dataspace2(4, dim2);

	IntType datatype(PredType::NATIVE_FLOAT);
	IntType datatype2(PredType::NATIVE_INT);

	DataSet dataset = file.createDataSet("data", datatype, dataspace);
	DataSet dataset2 = file.createDataSet("label", datatype2, dataspace2);


	FILE* image = fopen(img_loc, "rb");
	FILE* label = fopen(label_loc, "rb");

	if (image == NULL) {
		cerr << "No image file" << endl;
		return;
	}

	if (label == NULL) {
		cerr << "No label file" << endl;
		return;
	}

	uchar* image_data = new uchar[N * PIXEL];

	fseek(image, 16, SEEK_SET);

	if (fread(image_data, 1, N * PIXEL, image) != N * PIXEL) {
		cerr << "Incomplete read image" << endl;
		return;
	}

	uchar* label_data = new uchar[N];

	fseek(label, 8, SEEK_SET);

	if (fread(label_data, 1, N, label) != N) {
		cerr << "Incomplete read label" << endl;
		return;
	}

	float* proc_img = new float[N * PIXEL];

	for (int i = 0; i != N * PIXEL; ++i) {
		float tmp = static_cast<int>(image_data[i]) / static_cast<float>(255);

		proc_img[i] = tmp;
	}

	delete[] image_data;

	int* proc_label = new int[N];

	for (int i = 0; i != N; ++i) {
		proc_label[i] = static_cast<int>(label_data[i]);

	}

	delete[] label_data;

	dataset.write(proc_img, PredType::NATIVE_FLOAT);
	dataset2.write(proc_label, PredType::NATIVE_INT);

	delete[] proc_img;
	delete[] proc_label;

	return;
}

void write_cifar(const int n, const char* v[]) {
	if (n < 2) {
	  cerr << "name img_loc label_loc" << endl;
	  return;
	}

	const char* filename = v[1];
	const char* img_loc = v[2];

	FILE* img_label = fopen(img_loc, "rb");

	uchar* img_label_data = new uchar[N2 * 3074];
	float* img_data = new float[N2 * 3074];
	int* label_data = new int[N2];

	if (img_label == NULL) {
		cerr << "File not found" << endl;
		return;
	}

	if (fread(img_label_data, 1, N2 * 3074, img_label) != N2 * 3074) {
		cerr << "Incomplete reading" << endl;
		return;
	}

	fclose(img_label);

	uchar* local_img_label = img_label_data;
	float* local_img = img_data;

	for (int i = 0; i != N2 - 1; ++i) {

		label_data[i] = static_cast<int>(local_img_label[1]);

		local_img_label += 2;

		for (int j = 0; j != 1023; ++j) {

			local_img[j] = static_cast<int>(local_img_label[2048 + j]) / 256.0f;
			local_img[j + 1024] = static_cast<int>(local_img_label[1024 + j]) / 256.0f;
			local_img[j + 2048] = static_cast<int>(local_img_label[j]) / 256.0f;

		}

		local_img_label += 3072;
		local_img += 3072;

	}

	delete[] img_label_data;

	double img_mean = 0;

	for (int i = 0; i != (N2 * 3072) - 1; ++i) {
		img_mean += img_data[i];
	}

	cout << img_mean << endl;

	img_mean /= 500000 * 3072;

	cout << img_mean << endl;

	for (int i = 0; i != (N2 * 3072) - 1; ++i) {
		img_data[i] -= img_mean;
	}

	H5File file(filename, H5F_ACC_TRUNC);

	hsize_t dim[4]{N2, 3, 32, 32};

	hsize_t dim2[4]{N2, 1, 1, 1};

	DataSpace dataspace(4, dim);

	DataSpace dataspace2(4, dim2);

	IntType datatype(PredType::NATIVE_FLOAT);
	IntType datatype2(PredType::NATIVE_INT);

	DataSet dataset = file.createDataSet("data", datatype, dataspace);
	DataSet dataset2 = file.createDataSet("label", datatype2, dataspace2);

	cout << "WRITING" << endl;

	dataset.write(img_data, PredType::NATIVE_FLOAT);
	dataset2.write(label_data, PredType::NATIVE_INT);

	delete[] img_data;
	delete[] label_data;

	cout << "DONE WRITTING" << endl;

	return;
}

void write_cifar2(const int n, const char* v[]) {
	if (n < 2) {
	  cerr << "name img_loc label_loc" << endl;
	  return;
	}

	const char* filename = v[1];
	const char* img_loc = v[2];


	FILE* img_label = fopen(img_loc, "rb");

	uchar* img_label_data = new uchar[10000 * 3073];
	float* img_data = new float[10000 * 3072];
	int* label_data = new int[10000];

	if (img_label == NULL) {
		cerr << "File not found" << endl;
		return;
	}

	if (fread(img_label_data, 1, 10000 * 3073, img_label) != 10000 * 3073) {
		cerr << "Incomplete reading" << endl;
		return;
	}

	fclose(img_label);

	uchar* local_img_label = img_label_data;
	float* local_img = img_data;

	for (int i = 0; i != 10000; ++i) {

		label_data[i] = static_cast<int>(local_img_label[0]);

		for (int j = 0; j != 3071; ++j) {

			local_img[3071 - j] = static_cast<int>(local_img_label[j]) / 255.0f;

		}

		local_img_label += 3073;
		local_img += 3072;

	}

	delete[] img_label_data;

	H5File file(filename, H5F_ACC_TRUNC);

	hsize_t dim[4]{10000, 3, 32, 32};

	hsize_t dim2[4]{10000, 1, 1, 1};

	DataSpace dataspace(4, dim);

	DataSpace dataspace2(4, dim2);

	IntType datatype(PredType::NATIVE_FLOAT);
	IntType datatype2(PredType::NATIVE_INT);

	DataSet dataset = file.createDataSet("data", datatype, dataspace);
	DataSet dataset2 = file.createDataSet("label", datatype2, dataspace2);

	cout << "WRITING" << endl;

	dataset.write(img_data, PredType::NATIVE_FLOAT);
	dataset2.write(label_data, PredType::NATIVE_INT);

	delete[] img_data;
	delete[] label_data;

	cout << "DONE WRITTING" << endl;

	return;
}

void write_self_driving(const int n, const char** v) {

	ifstream fs(v[1]);

	char line[128];

	vector<float> steering_angle;
	vector<string> camera_location;

	if (fs.is_open()) {

		while (!fs.eof()) {

			for (int i = 0; i != 6; ++i) {

				fs.getline(line, 128, ',');

				if (i == 0 || i == 1 || i == 2) {

					camera_location.push_back(line);

				}

				if (i == 3) {

					steering_angle.push_back(atof(line));

				}

			}

			fs.getline(line, 128);

		}

		H5File file(v[2], H5F_ACC_TRUNC);

//		hsize_t dim[4]{3263, 3, 16, 32};
		hsize_t dim[4]{3263, 1, 16, 32};

		hsize_t dim2[4]{3263, 1, 1, 1};

		DataSpace dataspace(4, dim);

		DataSpace dataspace2(4, dim2);

		IntType datatype(PredType::NATIVE_FLOAT);
		IntType datatype2(PredType::NATIVE_FLOAT);

		DataSet dataset = file.createDataSet("data", datatype, dataspace);
		DataSet dataset2 = file.createDataSet("label", datatype2, dataspace2);

		cout << "WRITING" << endl;

		dataset2.write(steering_angle.data(), PredType::NATIVE_FLOAT);

//		float* data = new float[3 * 16 * 32 * 3263];
		float* data = new float[16 * 32 * 3263];

		vector<cv::Mat> colors;

		float* temp_ptr = data;

		for (int i = 0; i != steering_angle.size() - 1; ++i) {

			cv::Mat img = cv::imread(camera_location[i]);

			cv::resize(img, img, cv::Size(32, 16));

			cv::cvtColor(img, img, cv::COLOR_BGR2HSV);

			for (int j = 0; j != (32 * 16) - 1; ++j) {

//				temp_ptr[j] = static_cast<int>(img.data[j * 3]) / static_cast<float>(255);
//				temp_ptr[j + (32 * 16)] = static_cast<int>(img.data[1 + j * 3]) / static_cast<float>(255);

				temp_ptr[j] = static_cast<int>(img.data[1 + j * 3]) / static_cast<float>(255);

//				temp_ptr[j + (32 * 16 * 2)] = static_cast<int>(img.data[2 + j * 3]) / static_cast<float>(255);


			}

			cv::imshow("PICTURE", cv::Mat(16, 32, CV_32FC1, temp_ptr));
			cv::waitKey(1);

//			temp_ptr += 32 * 16 * 3;
			temp_ptr += 32 * 16;

		}


		dataset.write(data, PredType::NATIVE_FLOAT);

		delete[] data;

		file.close();

		fs.close();

	}

}

int main(const int argc, const char* argv[]) {


	write_self_driving(argc, argv);

	return 0;
}


