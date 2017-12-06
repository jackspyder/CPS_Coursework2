#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <ostream>
#include <iostream>
#include <thread>

//Define Global Variables
constexpr float softening = 1e-9f; //define softening value
constexpr float m = 1.0f; //define mass
constexpr float t_step = 0.001f; //define timestep
constexpr int n_iters = 10; //define number of iterations
constexpr int n_bodies = 10000; //define number of bodies

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

int main()
{
	//start chrono timer
	auto const startT = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> average;

	float* data = (float*)malloc(n_bodies * sizeof(Body));
	Body* planets = (Body*)data;

	//generate initial pos/vel data
	randomize_bodies(data, 6 * n_bodies);

	//loop for number of iterations
	for (int iter = 1; iter <= n_iters; iter++)
	{
		// Record start time
		auto const start = std::chrono::high_resolution_clock::now();

		//call body_force method to calculate forces
		body_force(planets, n_bodies);


		//loop to integration new postions
		for (auto i = 0; i < n_bodies; i++)
		{
			planets[i].x += planets[i].vx * t_step;
			planets[i].y += planets[i].vy * t_step;
			planets[i].z += planets[i].vz * t_step;
		}

		//calculate and display iteration number and iteration runtime.
		auto const finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		average += elapsed;
		std::cout << "Iteration: " << iter << " runtime: " << elapsed.count() << "ms" << std::endl;
	}
	//calculate total run time.
	auto const finishT = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> totalTime = finishT - startT;

	std::cout << "Total Run time for iterations 1 through " << n_iters << ": " << totalTime.count() << " seconds" << std::
		endl;
	std::cout << "Number of bodies calculated: " << n_bodies << std::endl;
	std::cout << "Average iteration runtime: " << average.count() / n_iters << std::endl;
	std::cout << "Iterations per second: " << n_iters / totalTime.count() << std::endl;

	//clear memory
	free(data);
}
