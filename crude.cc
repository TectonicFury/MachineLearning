#include <iostream>
#include <string>
#include <fstream>
#include "Desktop/pyAhmad/C++_Algs4/MLP_Class.h"

int main() {

	std::vector<std::vector<float>> vData;
	std::string str;
	std::string tempstr;
	std::vector<float> vTemp;

	while(std::getline(std::cin, str)) {
		for (unsigned long i = 0; i < str.size(); i++) {
			if (std::isdigit(str[i]) || str[i] == '.') {
				tempstr.append(1, str[i]);
			} else {
				vTemp.push_back(std::stof(tempstr));
				tempstr = "";
			}
		}
		vData.push_back(vTemp);
		vTemp.clear();
	}
	
	std::vector<int> M(4);
	M[0] = 1;
	M[1] = 25;
	M[2] = 25;
	M[3] = 17;
	Matrix<float, 1, 31> X; //crude feed rate
	Matrix<float, 17, 31> T; //target matrix of products
	for (int i = 0; i < T.rows(); i++) {
		for (int j = 0; j < T.cols(); j++) {
			T(i, j) = vData[i + 1][j];
		}
	}
	MLP mlp_crd(M, X, T, 100000);
	//MLP(std::vector<int> nodesPerLayer, Matrix<float, Dynamic, Dynamic, ColMajor> inputData, Matrix<float, Dynamic, Dynamic, ColMajor> targetMatrix, int totalTraining)
	auto start = std::chrono::steady_clock::now();
	mlp_crd.MLP_Train_Regression();
	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	unsigned long long totalTime = std::chrono::duration <double, std::milli> (diff).count();
	std::cout<<"total time = "<<totalTime<<" ms\n";
	std::vector<float> erV = mlp_crd.getErrV();
	
	std::ofstream dataFile;
	dataFile.open("errorCRD.txt");
	for (long long int i = 0; i < erV.size(); i++) {
		dataFile<<erV[i]<<std::endl;
	}
	dataFile.close();
}