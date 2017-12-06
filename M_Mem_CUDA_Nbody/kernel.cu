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

//Generate random body values
void randomize_bodies(float *data, const int n) {
	for (int i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f; //assign random float between 1 and -1
	}
}

//GPU Kernel that calculates forces on each body.
__global__
void body_force(Body *p, const int n) {
	//Work out grid size and use as index.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		//clear any previous forces.
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		for (int j = 0; j < n; j++) {
			//calculate distances between two bodies.
			const float dx = p[j].x - p[i].x;
			const float dy = p[j].y - p[i].y;
			const float dz = p[j].z - p[i].z;
			//square distances and add a softening factor
			const float dist_sqr = dx*dx + dy*dy + dz*dz + softening;
			const float inv_dist = rsqrtf(dist_sqr);
			const float inv_dist3 = inv_dist * inv_dist * inv_dist;

			//calculate the forces for each dimension on a body
			Fx += dx * inv_dist3;
			Fy += dy * inv_dist3;
			Fz += dz * inv_dist3;
		}

		//add the resulting velocities to the body.
		p[i].vx += t_step*Fx;
		p[i].vy += t_step*Fy;
		p[i].vz += t_step*Fz;
	}
}

int main()
{

	//start chrono timer
	auto const startT = std::chrono::high_resolution_clock::now();
	
	//calculate number of blocks
	int blocks = (n_bodies + block_size - 1) / block_size;

	//pointer to data object and allocate memory.
	float *data = (float*)malloc(n_bodies * sizeof(Body));

	//allocate host memory for h_planets pointer
	Body *h_planets = (Body*)malloc(n_bodies * sizeof(Body));

	//cast data to H-planets
	h_planets = reinterpret_cast<Body*>(data);

	//randomize body data
	randomize_bodies(data, 6 * n_bodies); 

	//pointer to device data
	float *d_data;

	//allocate Device memory
	cudaMalloc(&d_data, n_bodies * sizeof(Body));

	//pointer to d_planets and cast in d_data
	Body *d_planets = reinterpret_cast<Body*>(d_data);

	float average = 0;
	//loop for iterations here
	for (int j = 0; j<=n_iters; j++)
	{

		//cuda kernel timer
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//copy data from host to device memory.
		cudaMemcpy(d_data, data, n_bodies * sizeof(Body), cudaMemcpyHostToDevice);


		//initialize Kernel
		cudaEventRecord(start);//start cuda timer
		body_force << <blocks, block_size >> >(d_planets, n_bodies);
		cudaEventRecord(stop);//end cuda timer
		
		//wait for GPU to finish before acessing results
		cudaDeviceSynchronize();

		//copy data from device to host memory
		cudaMemcpy(data, d_data, n_bodies * sizeof(Body), cudaMemcpyDeviceToHost);

		//loop to integration new postions
		for (int i = 0; i < n_bodies; i++)
		{
			h_planets[i].x += h_planets[i].vx*t_step;
			h_planets[i].y += h_planets[i].vy*t_step;
			h_planets[i].z += h_planets[i].vz*t_step;
		}


		//calculate and display iteration number and iteration runtime.
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Iteration: " << j << " runtime: " << milliseconds << "ms" << std::endl;
		average += milliseconds;
		if (j == n_iters)
		{
			average = average / n_iters;
			std::cout << "Average effective bandwidth(MB/s): " << n_bodies * sizeof(Body) * 2 / average / 0x3E8 << std::endl;
		}
	}

	//calculate total run time.
	auto const finishT = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> totalTime = finishT - startT;
	std::cout << "Total Run time for iterations 1 through " << n_iters << ": " << totalTime.count() << " seconds" << std::endl;
	std::cout << "Number of bodies calculated: " << n_bodies << std::endl;
	std::cout << "Iterations per second: " << n_iters / totalTime.count() << std::endl;
	//clear memory
	cudaFree(d_data);
	cudaFree(d_planets);
	free(data);

	return 0;
}