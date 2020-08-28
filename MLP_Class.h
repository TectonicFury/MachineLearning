#ifndef MLP_CLASS_H
#define MLP_CLASS_H

#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <math.h>
#include <assert.h>
using namespace Eigen;

class MLP {
public:
	MLP(std::vector<int> nodesPerLayer, Matrix<float, Dynamic, Dynamic, ColMajor> inputData, Matrix<float, Dynamic, Dynamic, ColMajor> targetMatrix, int totalTraining);
	void MLP_Train_Regression();
	void MLP_Train_Classifier();
	Matrix<float, Dynamic, Dynamic> recallRegression(Matrix<float, Dynamic, Dynamic> inputData);
	Matrix<float, Dynamic, Dynamic> getY();
	Matrix<float, Dynamic, 1> getMEAN();
	Matrix<float, Dynamic, Dynamic> getSTDDEV();
	std::vector<Matrix<float, Dynamic, Dynamic>> getW();
	std::vector<float> getErrV();
private:
	Matrix<float, Dynamic, Dynamic, ColMajor> X;
	Matrix<float, Dynamic, Dynamic, ColMajor> XNormed;
	Matrix<float, Dynamic, Dynamic, ColMajor> t;
	std::vector<int> M;
	int T;
	std::vector<float> errorV;
	std::vector<Matrix<float, Dynamic, Dynamic>> w;
	std::vector<Matrix<float, Dynamic, 1>> a;
	std::vector<Matrix<float, Dynamic, 1>> δh;
	Matrix<float, Dynamic, Dynamic> y;
	unsigned long MSize;
	Matrix<float, Dynamic, 1> x_mean;
	Matrix<float, Dynamic, 1> stddev;
};
#endif