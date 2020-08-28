//#define NDEBUG
#include "MLP_Class.h"
#include <fstream>
//the number of columns in the weight matrix is equal to the number of nodes in a particular layer
float sigmoid(float x) {
	return 1.0 / (1 + exp(-3 * x));
}

float func1(float x) {
	return sin(2 * M_PI * x) + cos(4 * M_PI * x);
	//return x*x;
}

float func2(float x) {
	return cos(2 * M_PI * x);
}

MLP::MLP(std::vector<int> nodesPerLayer, Matrix<float, Dynamic, Dynamic, ColMajor> inputData, Matrix<float, Dynamic, Dynamic, ColMajor> targetMatrix, int totalTraining) {
		//M[0] is the number of features in the input
		//M[M.size() - 1] = N the number of output nodes
		assert(nodesPerLayer.size() > 2); //smallest MLP has 3 layers
		XNormed.resize(inputData.rows(), inputData.cols());
		t = targetMatrix;
		M = nodesPerLayer;
		T = totalTraining;
		MSize = M.size();
		w.resize(MSize - 1);
		a.resize(MSize - 1);
		δh.resize(MSize - 1);
		y.resize(M[MSize - 1], t.cols());
		//normalisation
		x_mean.resize(inputData.rows()); // this specifies the total number of features in the input
		stddev.resize(inputData.rows());
		
		for (int i = 0; i < x_mean.rows(); i++) {
			x_mean(i) = inputData.row(i).mean();
		}

		Matrix<float, Dynamic, 1> OnesM = Matrix<float, Dynamic, 1>::Ones(inputData.cols());

		for (int i = 0; i < stddev.rows(); i++) {
			auto a = inputData.block(i, 0, 1, inputData.cols()).transpose() - x_mean(i) * OnesM;

			auto b = a.transpose() * a / inputData.cols();
			stddev(i) = sqrt(b(0, 0));
		}

		for (int i = 0; i < XNormed.rows(); i++) {
			auto a = (inputData.block(i, 0, 1, inputData.cols()).transpose() - x_mean(i) * OnesM).transpose();
			auto b = a/stddev(i, 0);
			XNormed.block(i, 0, 1, inputData.cols()) = b;
		}

		X = XNormed;
		
		for (unsigned long i = 0; i < w.size();) { //initialisation of weights
			w[i] = Matrix<float, Dynamic, Dynamic>::Random(M[i] + 1, M[i + 1]);
			i++;
		}
		
		for (unsigned long i = 0; i < a.size(); i++) {
			//reserving space for the outputs of each layer and one for the bias input
			a[i] = Matrix<float, Dynamic, 1>(M[i] + 1); 
			a[i](0, 0) = -1; //setting the bias node of each layer to -1
		}

		for (unsigned long i = 1; i < MSize; i++) {
			δh[i - 1] = Matrix<float, Dynamic, 1> (M[i]); //the zeroth δh corresponds to the first hidden layer
		}	
	}
	
	//inference after regression training
Matrix<float, Dynamic, Dynamic> MLP::recallRegression(Matrix<float, Dynamic, Dynamic> inputData) {
		y.resize(M[MSize - 1], inputData.cols());
		Matrix<float, Dynamic, Dynamic> h;
		Matrix<float, Dynamic, Dynamic> inNormed(inputData.rows(), inputData.cols());
		
		Matrix<float, Dynamic, 1> OnesM = Matrix<float, Dynamic, 1>::Ones(inputData.cols());
		
		for (int i = 0; i < inNormed.rows(); i++) {
			auto a = (inputData.block(i, 0, 1, inputData.cols()).transpose() - x_mean(i) * OnesM).transpose();
			auto b = a/stddev(i);
			inNormed.block(i, 0, 1, inputData.cols()) = b;
		}
		
		for (int i = 0; i < inNormed.cols(); i++) {
			/**************** Forward pass ******************/
			a[0].block(1, 0, a[0].rows() - 1, 1) = inNormed.col(i); //setting the zeroth column of 'a' matrix to the input vector      
			for (unsigned long j = 0; j < MSize - 2; j++) {
				h = a[j].transpose() * w[j]; // get all the hk's at once
				std::cout<<"h\n";
				std::cout<<h<<"\n";
				auto b = h.unaryExpr(&sigmoid).transpose(); 
				a[j + 1].block(1, 0, M[j + 1], 1) = b;//filling the the inputs to the next hidden layer for a particular vector in X
			}
	
			//forward pass for the output layer
			h = a[MSize - 2].transpose() * w[MSize - 2]; // get all the hk's at once, modified for regression
			y.col(i) = h; //output vector
		}
		
		return y;
}
	
	//training for regression problem
void MLP::MLP_Train_Regression() {
		y.resize(M[MSize - 1], t.cols());
		unsigned long long int count = 0;
		float η = 0.025;
		int nTrngData = X.cols();
	 	Matrix<float, Dynamic, Dynamic> h;
		MatrixXf MOnes = MatrixXf::Ones(M[MSize - 1], 1);
		float error;
		//each input column of X will have first entry -1 for the input bias 

		for (int epoch = 0; epoch < T; epoch++) {
			//if (epoch > 0 && epoch % 100 == 0) η -= 0.01;
			for (int i = 0; i < nTrngData; i++) {
				/**************** Forward pass ******************/
				count++;
				a[0].block(1, 0, a[0].rows() - 1, 1) = X.col(i); //setting the zeroth column of 'a' matrix to the input vector
				
				for (unsigned long j = 0; j < MSize - 2; j++) {
					//printf("j = %lu a[%lu].rows() = %ld, w[%lu].rows() = %ld, w[%lu].cols() = %ld\n", j, j, a[j].rows(), j, w[j].rows(), j, w[j].cols());
					h = a[j].transpose() * w[j]; // get all the hk's at once
					//printf("h.cols() = %ld\n", h.cols());
					auto b = h.unaryExpr(&sigmoid).transpose(); 
					a[j + 1].block(1, 0, M[j + 1], 1) = b;//filling the the inputs to the next hidden layer for a particular vector in X
				}
				
				//forward pass for the output layer
				h = (a[MSize - 2].transpose() * w[MSize - 2]).transpose(); // get all the hk's at once, modified for regression
				//std::cout<<h<<" h\n";
				y.col(i) = h; //output vector
				//std::cout<<y.col(i)<<" ycoli\n";
				error = ((y.col(i) - t.col(i)).transpose() * (y.col(i) - t.col(i)))(0, 0);
				//printf("inreg\n");
				if (count%10000 == 0) {
					count = 0;
					error = ((y.col(i) - t.col(i)).transpose() * (y.col(i) - t.col(i)))(0, 0);
					errorV.push_back(sqrt(error));
				} 
				
				//printf("error = %f\n", error);
				/***************** Backpropagation phase ******************/

				//running once for the output layer, iterating over all the output nodes
				δh[MSize - 2] = y.col(i) - t.col(i); // one step calculation of all δο(κ).
			
				for (unsigned long j = MSize - 2; j >= 1; j--) {
					w[j] = w[j] - η * a[j] * δh[j].transpose(); //the whole matrix would get updated
					// need to backpropagate the errors to the previous layer, need to get δh[j - 1]
					for (int l = 0; l < δh[j - 1].rows(); l++) {
						auto b = w[j].block(l + 1, 0, 1, w[j].cols()) * δh[j];
						δh[j - 1](l) = a[j](l + 1, 0) * (1.0 - a[j](l + 1, 0)) * b(0, 0);
					}
				}
				w[0] = w[0] - η * a[0] * δh[0].transpose(); //updating the weights of the last layer
			}
			//printf("T = %d error = %f\n", epoch, error);
		}
		printf("total count = %llu\n", count);
}
	
	// training for classification problem
void MLP::MLP_Train_Classifier() {
		y.resize(M[MSize - 1], t.cols());
		float η = 0.25;
		int nTrngData = X.cols();
	 	Matrix<float, Dynamic, Dynamic> h;
		MatrixXf MOnes = MatrixXf::Ones(M[MSize - 1], 1);
	
		//each input column of X will have first entry -1 for the input bias 
		for (int epoch = 0; epoch < T; epoch++) {
			for (int i = 0; i < nTrngData; i++) {
				/**************** Forward pass ******************/
				a[0].block(1, 0, a[0].rows() - 1, 1) = X.col(i); //setting the zeroth column of 'a' matrix to the input vector
				for (unsigned long j = 0; j < MSize - 2; j++) {
					h = a[j].transpose() * w[j]; // get all the hk's at once
					auto b = h.unaryExpr(&sigmoid).transpose(); 
					a[j + 1].block(1, 0, M[j + 1], 1) = b;//filling the the inputs to the next hidden layer for a particular vector in X
				}
				//forward pass for the output layer
				h = a[MSize - 2].transpose() * w[MSize - 2]; // get all the hk's at once
				y.col(i) = h.unaryExpr(&sigmoid); //output vector
			
				/***************** Backpropagation phase ******************/

				//running once for the output layer, iterating over all the output nodes
				δh[MSize - 2] = (y.col(i) - t.col(i)) * (y.col(i).transpose() * (MOnes - y.col(i))); // one step calculation of all δο(κ).
			
				for (unsigned long j = MSize - 2; j >= 1; j--) {
					w[j] = w[j] - η * a[j] * δh[j].transpose(); //the whole matrix would get updated
					// need to backpropagate the errors to the previous layer, need to get δh[j - 1]
					for (int l = 0; l < δh[j - 1].rows(); l++) {
						auto b = w[j].block(l + 1, 0, 1, w[j].cols()) * δh[j];
						δh[j - 1](l) = a[j](l + 1, 0) * (1.0 - a[j](l + 1, 0)) * b(0, 0);
					}
				}
				w[0] = w[0] - η * a[0] * δh[0].transpose(); //updating the weights of the last layer
			}
		}
	}
	
Matrix<float, Dynamic, Dynamic> MLP::getY() {
		return y;
}
	
Matrix<float, Dynamic, 1> MLP::getMEAN() {
		return x_mean;
}
	
Matrix<float, Dynamic, Dynamic> MLP::getSTDDEV() {
		return stddev;
}
	
std::vector<Matrix<float, Dynamic, Dynamic>> MLP::getW() {
		return w;
}

std::vector<float> MLP::getErrV() {
	return errorV;
}

int main() {
	std::vector<int> M(3); //has information about the number of inputs, hidden nodes and output nodes in each of the layers
	M[0] = 1; //two input features
	M[1] = 10; //one output node, no hidden layers
	M[2] = 2; //we have to change this also if we have multiple outputs
	//M[3] = 1;

	Matrix<float, 1, 40> X;
	for (int i = 0; i < X.cols(); i++) {
		X(i) = (float)(i + 1)/X.cols();
	}

	Matrix<float, Dynamic, Dynamic> Y;
	Y.resize(M[M.size() - 1], X.cols());
	Y.row(0) = X.unaryExpr(&func1);

	Y.row(1) = X.unaryExpr(&func2);
	/*for (int i = 0; i < X.cols(); i++) {
		std::cout<<X(i)<<"  "<<Y(i)<<"\n";
	}*/
	MLP mlpReg(M, X, Y, 1000000);
	
	std::cout<<"mean\n";
	std::cout<<mlpReg.getMEAN()<<"\n";
	std::cout<<"stddev\n";
	std::cout<<mlpReg.getSTDDEV()<<"\n";
	
	auto start = std::chrono::steady_clock::now();
	mlpReg.MLP_Train_Regression();
	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;
	unsigned long long totalTime = std::chrono::duration <double, std::milli> (diff).count();
	std::cout<<"total time = "<<totalTime<<"\n";
	std::vector<float> erV = mlpReg.getErrV();
	
	std::ofstream dataFile;
	dataFile.open("errorV.txt");
	for (unsigned long i = 0; i < erV.size(); i++) {
		dataFile<<erV[i]<<std::endl;
	}
	dataFile.close();
	
	std::vector<Matrix<float, Dynamic, Dynamic>> w = mlpReg.getW();
	dataFile.open("weights.txt");
	for (unsigned long i = 0; i < w.size(); i++) {
		dataFile<<w[i]<<"\n";
	}
	dataFile.close();
	//recall testing Alhamdulillaah works well
	Matrix<float, Dynamic, Dynamic> Xrecall(1, 1), Yrecall(1, 1);
	std::minstd_rand engine(time(NULL));
	std::uniform_real_distribution<float> dist(0, 1);
	for (int i = 0; i < Xrecall.cols(); i++) {
		Xrecall(i) = dist(engine);
	}
	Xrecall(0) = 0.0;
	
	Matrix<float, Dynamic, Dynamic> Yactual = Xrecall.unaryExpr(&func1);
	Yrecall = mlpReg.recallRegression(Xrecall);
	std::cout<<Yrecall<<"\n";
	std::cout<<Yactual<<"\n";
}