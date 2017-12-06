#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>

//Define Global Variables
constexpr float softening = 1e-9f;  //define softening value
constexpr float m = 1.0f;			//define mass
constexpr float t_step = 0.001f;	//define timestep
constexpr int n_iters = 10;			//define number of iterations
constexpr int n_bodies = 200000;	//define number of bodies
constexpr int block_size = 128;		//define block size

//define body struct using float4
typedef struct { float4 *pos, *vel; } BodySystem;

//Generate random body values
void randomize_bodies(float *data, const int n) {
	for (auto i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f; //assign random float between 1 and -1
	}
}

//GPU Kernel that calculates forces on each body.
__global__
void body_force(float4 *p, float4 *v, const int n) {
	//Work out grid size and use as index.
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		//clear any previous forces.
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		for (auto j = 0; j < n; j++) {
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
		v[i].x += t_step*Fx;
		v[i].y += t_step*Fy;
		v[i].z += t_step*Fz;
	}
}


int main() {

	//start chrono timer
	auto const startT = std::chrono::high_resolution_clock::now();

	//pre calculate bytes for easy memory allocation
	const int bytes = 2 * n_bodies * sizeof(float4);

	//pointer and memory alocation for data, declare planets
	float *data = (float*)malloc(bytes);
	const BodySystem planets = { (float4*)data, ((float4*)data) + n_bodies };

	//randomise body data
	randomize_bodies(data, 8 * n_bodies); // Init pos / vel data

	//pointer for device data and allocate device memory
	float *d_data;
	cudaMalloc(&d_data, bytes);
	const BodySystem d_planets = { (float4*)d_data, ((float4*)d_data) + n_bodies };

	//calculate number of thread blocks
	auto n_blocks = (n_bodies + block_size - 1) / block_size;
	
	float average = 0;

	//main loop
	for (auto iter = 1; iter <= n_iters; iter++) 
	{
		//cuda kernel timer
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//copy from host ot device memory
		cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);

		//time and execute body_force kernel
		cudaEventRecord(start);//start cuda timer
		body_force << <n_blocks, block_size >> >(d_planets.pos, d_planets.vel, n_bodies);
		cudaEventRecord(stop);//end cuda timer

		//return data from device
		cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);

		// integrate position data
		for (auto i = 0; i < n_bodies; i++) 
		{ 
			planets.pos[i].x += planets.vel[i].x*t_step;
			planets.pos[i].y += planets.vel[i].y*t_step;
			planets.pos[i].z += planets.vel[i].z*t_step;
		}

		//calculate and display iteration number and iteration runtime.
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Iteration: " << iter << " runtime: " << milliseconds << "ms" << std::endl;
		average += milliseconds;
		if (iter == n_iters)
		{
			//display bandwidth on last lteration only, to avoid spam.
			average = average / n_iters;
			std::cout << "Average effective bandwidth(MB/s): " << n_bodies * sizeof(BodySystem) * 2 / average / 0x3E8 << std::endl;
		}
		
	}
	//calculate total run time.
	auto const finishT = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> totalTime = finishT - startT;
	std::cout << "Total Run time for iterations 1 through " << n_iters << ": " << totalTime.count() << " seconds" << std::endl;
	std::cout << "Number of bodies calculated: " << n_bodies << std::endl;
	std::cout << "Iterations per second: " << n_iters / totalTime.count() << std::endl;
	//clear data
	cudaFree(d_data);
	free(data);
}