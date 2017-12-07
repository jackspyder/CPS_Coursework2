#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ostream>
#include <iostream>
#include <thread>
#include <omp.h>
#include <chrono>

//Define Global Variables
constexpr float softening = 1e-9f; //define softening value
constexpr float m = 1.0f; //define mass
constexpr float t_step = 0.001f; //define timestep
constexpr int n_iters = 10; //define number of iterations
constexpr int n_bodies = 12500; //define number of bodies

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

//Calculate forces on a body
void body_force(Body* p, const int n)
{
	//Initiate openMP
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < n; i++)
	{
		//clear any previous forces
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
			const float inv_dist = m / sqrtf(dist_sqr);
			const float inv_dist3 = inv_dist * inv_dist * inv_dist;

			//calculate the forces for each dimension on a body
			Fx += dx * inv_dist3;
			Fy += dy * inv_dist3;
			Fz += dz * inv_dist3;
		}

		//add the resulting velocities to the body
		p[i].vx += t_step * Fx;
		p[i].vy += t_step * Fy;
		p[i].vz += t_step * Fz;
	}
}

void body_position(const int n, Body *p)
{
#pragma omp parallel for schedule(guided)
	for (int i = 0; i < n_bodies; i++)
	{
		p[i].x += p[i].vx * t_step;
		p[i].y += p[i].vy * t_step;
		p[i].z += p[i].vz * t_step;
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

	int bytes = n_bodies * sizeof(Body);

	float* data = (float*)malloc(bytes);
	Body* planets = (Body*)data;

	//generate initial pos/vel data
	randomize_bodies(data, 6 * n_bodies);

	//loop for number of iterations
	for (int iter = 1; iter <= n_iters; iter++)
	{
		//cudaEventRecord(looptime_start);//start cuda looptimer
		auto const looptime_start = std::chrono::high_resolution_clock::now();
		if (iter == 1) { initial = looptime_start - runtime_start; }
		//body_force call and timers
		auto const forcetime_start = std::chrono::high_resolution_clock::now();
		body_force(planets, n_bodies);
		auto const forcetime_stop = std::chrono::high_resolution_clock::now();
		forcetime_total += forcetime_stop - forcetime_start;
		
		//body_position call and timers
		auto const positiontime_start = std::chrono::high_resolution_clock::now();
		body_position(n_bodies, planets);
		auto const positiontime_stop = std::chrono::high_resolution_clock::now();
		positontime_total += positiontime_stop - positiontime_start;

		//calculate and display iteration number and iteration runtime.
		auto const looptime_stop = std::chrono::high_resolution_clock::now();
		looptime = looptime_stop - looptime_start;
		looptime_total += looptime_stop - looptime_start;
		std::cout << "Iteration: " << iter << " runtime: " << looptime.count() << " seconds" << std::endl;
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

	//clear memory
	free(data);
}