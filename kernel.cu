#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>

#define BLOCKSIZE 128
#define GRIDSIZE 64
#define VECTORSIZE 3
#define FUNCTIONQUANTITY 2
#define ENDCRITERIA 0.1

typedef double(*func)(double*);

typedef struct {
	double mins[VECTORSIZE];
	double maxs[VECTORSIZE];
} Box;

double diam(Box &box) {
	double diam = 0;
	for (int i = 0; i < VECTORSIZE; i++)
		diam = fmax(diam, abs(box.maxs[i] - box.mins[i]));
	return diam;
}

std::pair<Box, Box> split(Box &box) {
	std::pair<Box, Box> r;
	int maxIndex = 0;
	double max = abs(box.maxs[0] - box.mins[0]);
	for (int i = 0; i < VECTORSIZE; i++) {
		double cur = abs(box.maxs[i] - box.mins[i]);
		if (cur > max){
			maxIndex = i;
			max = cur;
		}
		r.first.mins[i] = box.mins[i];
		r.first.maxs[i] = box.maxs[i];
		r.second.mins[i] = box.mins[i];
		r.second.maxs[i] = box.maxs[i];
	}
	double border = box.mins[maxIndex] + abs(box.maxs[maxIndex] - box.mins[maxIndex]) / 2;
	r.first.maxs[maxIndex] = border;
	r.second.mins[maxIndex] = border;
	return r;
}
//Function description
__device__ double f1(double* vector) {
	return (vector[0] + 3)*(vector[0] + 3) + (vector[1])*(vector[1]) - 16;
}

__device__ double f2(double* vector) {
	return (vector[0]-3)*(vector[0]-3) + (vector[1])*(vector[1])- 16;
}

__device__ double f3(double* vector) {
	return (vector[0])*(vector[0]) + (vector[1])*(vector[1]) - 9;
}

template <func... Functions>
__global__ void addKernel(Box box, double* maxout, double* minout, double* retmax, double* retmin)
{
	__shared__ double smax[FUNCTIONQUANTITY*BLOCKSIZE];
	__shared__ double smin[FUNCTIONQUANTITY*BLOCKSIZE];

	int dim = pow(double(BLOCKSIZE * GRIDSIZE), 1 / double(VECTORSIZE));
	constexpr func table[] = { Functions... };
	double vec[VECTORSIZE];
	int threadNum = (threadIdx.x + blockIdx.x * blockDim.x);
	for (int i = 0; i < VECTORSIZE; i++) {
		vec[i] = box.mins[i] + (abs(box.maxs[i] - box.mins[i]) / dim) * (threadNum % dim);
		threadNum /= dim;
	}
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		func fun = table[i];
		smax[i*blockDim.x + threadIdx.x] = fun(vec);
		smin[i*blockDim.x + threadIdx.x] = fun(vec);
	}
	__syncthreads();
	int s = blockDim.x >> 1;
	while (s != 0) {
		if (threadIdx.x < s) {
			int su = threadIdx.x + s;
			for (int i = 0; i < FUNCTIONQUANTITY; i++) {
				smax[i*blockDim.x + threadIdx.x] = fmax(smax[i*blockDim.x + threadIdx.x], smax[i*blockDim.x + su]);
				smin[i*blockDim.x + threadIdx.x] = fmin(smin[i*blockDim.x + threadIdx.x], smin[i*blockDim.x + su]);
			}
		}
		__syncthreads();
		s >>= 1;
	}
	__syncthreads();
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		maxout[i*gridDim.x + blockIdx.x] = smax[i*blockDim.x];
		minout[i*gridDim.x + blockIdx.x] = smin[i*blockDim.x];
	}
	__syncthreads();
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		if (threadIdx.x < gridDim.x) {
			smax[i*blockDim.x + threadIdx.x] = maxout[i*gridDim.x + threadIdx.x];
			smin[i*blockDim.x + threadIdx.x] = minout[i*gridDim.x + threadIdx.x];
		}
		else {
			smax[i*blockDim.x + threadIdx.x] = maxout[i*gridDim.x];
			smin[i*blockDim.x + threadIdx.x] = minout[i*gridDim.x];
		}
	}
	__syncthreads();
	s = blockDim.x >> 1;
	while (s != 0) {
		if (threadIdx.x < s) {
			int su = threadIdx.x + s;
			for (int i = 0; i < FUNCTIONQUANTITY; i++) {
				smax[i*blockDim.x + threadIdx.x] = fmax(smax[i*blockDim.x + threadIdx.x], smax[i*blockDim.x + su]);
				smin[i*blockDim.x + threadIdx.x] = fmin(smin[i*blockDim.x + threadIdx.x], smin[i*blockDim.x + su]);
			}
		}
		__syncthreads();
		s >>= 1;
	}
	for (int i = 0; i < FUNCTIONQUANTITY; i++) {
		retmax[i] = smax[i*blockDim.x];
		retmin[i] = smin[i*blockDim.x];
	}
	__syncthreads();
}

bool checkAxes(double* minsMaxes, Box& p, int numberOfAxes, int*axes) {
	bool flag = true;
	for (int i = 0; i < numberOfAxes; i++) {
		if (!((p.mins[axes[i] - 1] > minsMaxes[i * 2]) && (p.maxs[axes[i] - 1] < minsMaxes[i * 2 + 1]))) {
			flag = false;
			break;
		}
	}
	return flag;
}

int main(int argc, char** argv)
{
	unsigned int start_time = clock();
	dim3 dimblock(BLOCKSIZE);
	dim3 dimgrid(GRIDSIZE);
	//Main box borders
	double mins[VECTORSIZE] = { -15,-15,-15};
	double maxs[VECTORSIZE] = { 15,15,15};

	Box box;
	for (int i = 0; i < VECTORSIZE; i++) {
		box.mins[i] = mins[i];
		box.maxs[i] = maxs[i];
	}
	std::vector<Box> temp;
	std::vector<Box> main;
	std::vector<Box> I;
	std::vector<Box> E;
	main.push_back(box);
	double curD = diam(box);
	double *retmin, *retmax, *rmin, *rmax;
	cudaMalloc((void**)&retmin, FUNCTIONQUANTITY * sizeof(double));
	cudaMalloc((void**)&retmax, FUNCTIONQUANTITY * sizeof(double));
	rmin = (double*)malloc(sizeof(double) * FUNCTIONQUANTITY);
	rmax = (double*)malloc(sizeof(double) * FUNCTIONQUANTITY);
	double* maxout, double* minout;
	int* sync;
	cudaMalloc((void**)&maxout, FUNCTIONQUANTITY*dimblock.x * sizeof(double));
	cudaMalloc((void**)&minout, FUNCTIONQUANTITY*dimblock.x * sizeof(double));
	cudaMalloc((void**)&sync, dimblock.x * sizeof(int));
	while (curD > ENDCRITERIA && main.size() > 0) {
		for (auto p : main) {
			//Place function names in template
			addKernel <f1,f2> <<<dimgrid, dimblock>> > (p, maxout,minout,retmax,retmin);
			cudaMemcpy(rmax, retmax, FUNCTIONQUANTITY * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(rmin, retmin, FUNCTIONQUANTITY * sizeof(double), cudaMemcpyDeviceToHost);
			double max = rmax[0];
			double min = rmin[0];
			for (int i = 1; i < FUNCTIONQUANTITY; i++) {
				max = fmax(rmax[i - 1], rmax[i]);
				min = fmax(rmin[i - 1], rmin[i]);
			}
			if (min > 0) {
				E.push_back(p);
				continue;
			}
			if (max < 0) {
				I.push_back(p);
				continue;
			}
			std::pair<Box, Box> sp = split(p);
			temp.push_back(sp.first);
			temp.push_back(sp.second);
			curD = diam(sp.first);
		}
		std::cout << "Main size: " << main.size() << " Cur diam: " << curD << "\n";
		main.clear();
		main.insert(main.begin(), temp.begin(), temp.end());
		temp.clear();
	}
	cudaFree(retmin);
	cudaFree(retmax);
	cudaFree(maxout);
	cudaFree(minout);
	free(rmax);
	free(rmin);
	unsigned int end_time = clock();
	unsigned int search_time = end_time - start_time;
	std::cout << search_time << "\n";
	std::ofstream myfile;
		myfile.open("out.txt");
		myfile << VECTORSIZE << "\n";
		for (int i = 0; i < VECTORSIZE; i++) {
			myfile << mins[i] << " " << maxs[i] << "\n";
		}
		myfile << main.size() << "\n";
		for (auto p : main) {
			for (int i = 0; i < VECTORSIZE; i++) {
				myfile << p.mins[i] << " " << p.maxs[i] << " ";
			}
			myfile << "\n";
		}
		myfile << I.size() << "\n";
		for (auto p : I) {
			for (int i = 0; i < VECTORSIZE; i++) {
				myfile << p.mins[i] << " " << p.maxs[i] << " ";
			}
			myfile << "\n";
		}
		myfile << E.size() << "\n";
		for (auto p : E) {
			for (int i = 0; i < VECTORSIZE; i++) {
				myfile << p.mins[i] << " " << p.maxs[i] << " ";
			}
			myfile << "\n";
		}
		myfile.close();
	if (argc >= 6) {
		int n1 = atoi(argv[1]);
		int n2 = atoi(argv[2]);
		if (argc % 3 != 0) {
			printf("Wrong arguments, cant assemble 2d output\n");
			return;
		}
		int numberOfAxes = (argc - 3) / 3;
		if (VECTORSIZE - numberOfAxes != 2) {
			printf("Wrong arguments, cant assemble 2d output\n");
			return;
		}
		int* axes = (int*)malloc(numberOfAxes * sizeof(int));
		double* minsMaxes = (double*)malloc(numberOfAxes * 2 * sizeof(double));
		for (int i = 0; i < numberOfAxes; i++) {
			axes[i] = atoi(argv[3 + i * 3]);
			minsMaxes[i*2] = atof(argv[3 + i * 3 + 1]);
			minsMaxes[i * 2 + 1] = atof(argv[3 + i * 3 + 2]);
		}
		myfile.open("out2d.txt");
		myfile << 2 << "\n";
		myfile << mins[n1-1] << " " << maxs[n1-1] << "\n";
		myfile << mins[n2 - 1] << " " << maxs[n2 - 1] << "\n";
		int mainSize = 0;
		for (auto p : main) {
			if (checkAxes(minsMaxes,p,numberOfAxes,axes)) {
				mainSize++;
			}
		}
		myfile << mainSize << "\n";
		for (auto p : main) {
			if (checkAxes(minsMaxes, p, numberOfAxes, axes)) {
				myfile << p.mins[n1 - 1] << " " << p.maxs[n1 - 1] << " ";
				myfile << p.mins[n2 - 1] << " " << p.maxs[n2 - 1] << " ";
				myfile << "\n";
			}
		}
		int ISize = 0;
		for (auto p : I) {
			if (checkAxes(minsMaxes, p, numberOfAxes, axes)) {
				ISize++;
			}
		}
		myfile << ISize << "\n";
		for (auto p : I) {
			if (checkAxes(minsMaxes, p, numberOfAxes, axes)) {
				myfile << p.mins[n1 - 1] << " " << p.maxs[n1 - 1] << " ";
				myfile << p.mins[n2 - 1] << " " << p.maxs[n2 - 1] << " ";
				myfile << "\n";
			}
		}
		int ESize = 0;
		for (auto p : E) {
			if (checkAxes(minsMaxes, p, numberOfAxes, axes)) {
				ESize++;
			}
		}
		myfile << ESize << "\n";
		for (auto p : E) {
			if (checkAxes(minsMaxes, p, numberOfAxes, axes)) {
				myfile << p.mins[n1 - 1] << " " << p.maxs[n1 - 1] << " ";
				myfile << p.mins[n2 - 1] << " " << p.maxs[n2 - 1] << " ";
				myfile << "\n";
			}
		}
		myfile.close();
	}
    return 0;
}
