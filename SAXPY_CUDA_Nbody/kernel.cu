#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>

//Define Global Variables
constexpr float softening = 1e-9f; //define softening value
constexpr float t_step = 0.001f; //define timestep
constexpr int n_iters = 10; //define number of iterations
constexpr int n_bodies = 200000; //define number of bodies
constexpr int block_size = 128; //define block size

								//Define Body structure
typedef struct
{
	float x, y, z, vx, vy, vz;
} Body;

//Generate random body values
void randomize_bodies(float* data, const int n)
{
	for (int i = 0; i < n; i++)
	{
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f; //assign random float between 1 and -1
	}
}

//GPU Kernel that calculates forces on each body.
__global__

void body_force(Body* p, const int n)
{
	//Work out grid size and use as index.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
	{
		//clear any previous forces.
		float Fx = 0.0f;
		float Fy = 0.0f;
		float Fz = 0.0f;

		for (int j = 0; j < n; j++)
		{
			//calculate distances between two bodies.
			const float dx = p[j].x - p[i].x;
			const float dy = p[j].y - p[i].y;
			const float dz = p[j].z - p[i].z;
			//square distances and add a softening factor
			const float dist_sqr = dx * dx + dy * dy + dz * dz + softening;
			const float inv_dist = rsqrtf(dist_sqr);
			const float inv_dist3 = inv_dist * inv_dist * inv_dist;

			//calculate the forces for each dimension on a body
			Fx += dx * inv_dist3;
			Fy += dy * inv_dist3;
			Fz += dz * inv_dist3;
		}

		//add the resulting velocities to the body.
		p[i].vx += t_step * Fx;
		p[i].vy += t_step * Fy;
		p[i].vz += t_step * Fz;
	}
}
__global__
void body_position(const int n, Body* p)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n)
	{
		p[i].x += p[i].vx * t_step;
		p[i].y += p[i].vy * t_step;
		p[i].z += p[i].vz * t_step;
	}
	
	
}

int main()
{
	//start chrono timer
	auto const startT = std::chrono::high_resolution_clock::now();

	//calculate number of thread blocks
	int blocks = (n_bodies + block_size - 1) / block_size;

	//Pointers for data and planets.
	float* data;
	Body* planets;

	//allocate unified memory for data and planets
	cudaMallocManaged(&data, n_bodies * sizeof(Body));
	cudaMallocManaged(&planets, n_bodies * sizeof(Body));

	//cast data to planets
	planets = reinterpret_cast<Body*>(data);

	//generate initial pos/vel data
	randomize_bodies(data, 6 * n_bodies);
	
	//loop for number of iterations
	for (int j = 1; j <= n_iters; j++)
	{
		//cuda kernel timer
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		//initialize Kernel
		cudaEventRecord(start);//start cuda timer
		body_force << <blocks, block_size >> >(planets, n_bodies);
		
				
		//Initialize Position calculation Kernel
		body_position<<<blocks, block_size>>>(n_bodies, planets);
		cudaEventRecord(stop);//end cuda timer

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
				
		std::cout << "Iteration: " << j << " runtime: " << milliseconds << "ms" << std::endl;
		
	}

	//calculate total run time.
	auto const finishT = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> totalTime = finishT - startT;
	std::cout << "Total Run time for iterations 1 through " << n_iters << ": " << totalTime.count() << " seconds" << std::
		endl;
	std::cout << "Number of bodies calculated: " << n_bodies << std::endl;
	std::cout << "Iterations per second: " << n_iters / totalTime.count() << std::endl;
	//clear memory
	cudaFree(planets);
	cudaFree(data);

	return 0;
}
