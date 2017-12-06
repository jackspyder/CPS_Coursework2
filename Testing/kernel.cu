#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>

//Define Global Variables
constexpr float softening = 1e-9f;  //define softening value
constexpr float t_step = 0.001f;	//define timestep
constexpr int n_iters = 500;			//define number of iterations
constexpr int n_bodies = 65572;		//define number of bodies
constexpr int block_size = 128;		//define block size
									//Define Body structure
typedef struct { float x, y, z, vx, vy, vz; } Body;

//Initialise random body values
void randomize_bodies(float *data, const int n) {
	for (auto i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f; //assign random float between 1 and -1
	}
}

//GPU Kernel that calculates forces on each body.
__global__
void body_force(Body *p, const int n) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		auto Fx = 0.0f; auto Fy = 0.0f; auto Fz = 0.0f;

		for (auto j = 0; j < n; j++) {
			const auto dx = p[j].x - p[i].x;
			const auto dy = p[j].y - p[i].y;
			const auto dz = p[j].z - p[i].z;
			const auto dist_sqr = dx*dx + dy*dy + dz*dz + softening;
			const auto inv_dist = rsqrtf(dist_sqr);
			const auto inv_dist3 = inv_dist * inv_dist * inv_dist;

			Fx += dx * inv_dist3;
			Fy += dy * inv_dist3;
			Fz += dz * inv_dist3;
		}

		p[i].vx += t_step*Fx;
		p[i].vy += t_step*Fy;
		p[i].vz += t_step*Fz;
	}
}

int main(void)
{

	auto const start = std::chrono::high_resolution_clock::now();
	//have my n_bodies
	int blocks = (n_bodies + block_size - 1) / block_size;

	//reference data object and reserve memory.
	float *data;
	Body *planets;
	cudaMallocManaged(&data, n_bodies * sizeof(Body));
	cudaMallocManaged(&planets, n_bodies * sizeof(Body));

	planets = reinterpret_cast<Body*>(data);
	randomize_bodies(data, 6 * n_bodies); // Init pos / vel data

	//loop for iterations here
	for (int i = 0; i<n_iters; i++)
	{

		//cuda kernel timer
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);


		//initialize Kernel
		cudaEventRecord(start);//start cuda timer
		body_force << <blocks, block_size >> >(planets, n_bodies);
		cudaEventRecord(stop);//end cuda timer
							  //wait for GPU to finish before acessing results
		cudaDeviceSynchronize();

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Effective Bandwidth (GB/s): " << n_bodies * 4 * 3 / milliseconds / 1e6 << std::endl;

		for (int i = 0; i < n_bodies; i++)
		{ // integrate position
			planets[i].x += planets[i].vx*t_step;
			planets[i].y += planets[i].vy*t_step;
			planets[i].z += planets[i].vz*t_step;
		}
	}

	auto const finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> totalTime = finish - start;

	std::cout << "Total Run time for iterations 1 through " << n_iters << ": " << totalTime.count() << " seconds" << std::endl;
	cudaFree(planets);
	cudaFree(data);

	return 0;
}