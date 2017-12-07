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

								//define body struct using float4
typedef struct
{
	float3 *pos, *vel;
} Body;

//Generate random body values
void randomize_bodies(float *data, const int n)
{
	for (int i = 0; i < n; i++)
	{
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f; //assign random float between 1 and -1
	}
}

//GPU Kernel that calculates forces on each body.
__global__

void body_force(float3 *p, float3 *v, const int n)
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
		v[i].x += t_step * Fx;
		v[i].y += t_step * Fy;
		v[i].z += t_step * Fz;
	}
}

void body_position(const int n, float3* p, float3* v)
{
	for (int i = 0; i < n_bodies; i++)
	{
		p[i].x += v[i].x * t_step;
		p[i].y += v[i].y * t_step;
		p[i].z += v[i].z * t_step;
	}


}

int main()
{
	//Declare timer events.
	auto const runtime_start = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> runtime_total;
	std::chrono::duration<double> looptime_total;
	std::chrono::duration<double> forcetime_total;
	std::chrono::duration<double> positontime_total;
	std::chrono::duration<double> looptime;
	std::chrono::duration<double> initial;

	int bytes = 2 * n_bodies * sizeof(float3);

	//calculate number of thread blocks
	int blocks = (n_bodies + block_size - 1) / block_size;

	//Pointers for data and planets.
	float* data;

	cudaMallocManaged(&data, bytes);

	Body planets = { (float3*)data, ((float3*)data) + n_bodies };



	//allocate unified memory for data and planets

	cudaMallocManaged(&planets.pos, bytes);
	cudaMallocManaged(&planets.vel, bytes);

	//generate initial pos/vel data
	randomize_bodies(data, 5 * n_bodies);

	//loop for number of iterations
	for (int j = 1; j <= n_iters; j++)
	{
		//cudaEventRecord(looptime_start);//start cuda looptimer
		auto const looptime_start = std::chrono::high_resolution_clock::now();
		if (j == 1) { initial = looptime_start - runtime_start; }
		//body_force call and timers
		auto const forcetime_start = std::chrono::high_resolution_clock::now();
		body_force << <blocks, block_size >> >(planets.pos, planets.vel, n_bodies);
		cudaDeviceSynchronize();
		auto const forcetime_stop = std::chrono::high_resolution_clock::now();
		forcetime_total += forcetime_stop - forcetime_start;


		//body_position call and timers
		auto const positiontime_start = std::chrono::high_resolution_clock::now();
		body_position(n_bodies, planets.pos, planets.vel);
		auto const positiontime_stop = std::chrono::high_resolution_clock::now();
		positontime_total += positiontime_stop - positiontime_start;

		//calculate and display iteration number and iteration runtime.
		auto const looptime_stop = std::chrono::high_resolution_clock::now();
		looptime = looptime_stop - looptime_start;
		looptime_total += looptime_stop - looptime_start;
		std::cout << "Iteration: " << j << " runtime: " << looptime.count() << " seconds" << std::endl;

	}

	//calculate and display run statistics
	auto const runtime_stop = std::chrono::high_resolution_clock::now();
	runtime_total = runtime_stop - runtime_start;
	std::cout << "Number of bodies calculated: " << n_bodies << std::endl;
	std::cout << "Total Run time: " << runtime_total.count() << " seconds" << std::endl;
	std::cout << "Initialize time: " << initial.count() << " seconds" << std::endl;
	std::cout << "Execution time: " << runtime_total.count() - initial.count() << " seconds" << std::endl;
	std::cout << "Average iteration time: " << looptime_total.count() / n_iters << std::endl;
	std::cout << "Iterations per second: " << n_iters / looptime_total.count() << std::endl;
	std::cout << "Force Bandwidth (MB/s): " << 2 * bytes / (forcetime_total.count() / n_iters) / 1000000 << std::endl;
	std::cout << "Positon Bandwidth (GB/s): " << 2 * bytes / (positontime_total.count() / n_iters) / 1e+9 << std::endl;

	return 0;
}
