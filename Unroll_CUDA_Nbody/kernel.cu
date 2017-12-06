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
constexpr int n_iters = 100;			//define number of iterations
constexpr int n_bodies = 100000;		//define number of bodies
constexpr int block_size = 128;		//define block size

//define body struct using float4
typedef struct { float4 *pos, *vel; } BodySystem;

//Initialise random body values
void randomize_bodies(float *data, const int n) {
	for (auto i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f; //assign random float between 1 and -1
	}
}

__global__
void bodyForce(float4 *p, float4 *v, const int n) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		auto Fx = 0.0f; auto Fy = 0.0f; auto Fz = 0.0f;

		for (auto j = 0; j < n; j++) {
			const auto dx = p[j].x - p[i].x;
			const auto dy = p[j].y - p[i].y;
			const auto dz = p[j].z - p[i].z;
			const auto dist_sqr = dx*dx + dy*dy + dz*dz + softening;
			const auto inv_dist = m / sqrtf(dist_sqr);
			const auto inv_dist3 = inv_dist * inv_dist * inv_dist;

			Fx += dx * inv_dist3;
			Fy += dy * inv_dist3;
			Fz += dz * inv_dist3;
		}

		v[i].x += t_step*Fx;
		v[i].y += t_step*Fy;
		v[i].z += t_step*Fz;
	}
}


int main() {

	const int bytes = 2 * n_bodies * sizeof(float4);
	auto *buf = (float*)malloc(bytes);
	const BodySystem p = { (float4*)buf, ((float4*)buf) + n_bodies };

	randomize_bodies(buf, 8 * n_bodies); // Init pos / vel data

	float *d_buf;
	cudaMalloc(&d_buf, bytes);
	const BodySystem d_p = {(float4*)d_buf, ((float4*)d_buf) + n_bodies};
	
	auto n_blocks = (n_bodies + block_size - 1) / block_size;
	std::chrono::duration<double> totalTime;

	for (auto iter = 1; iter <= n_iters; iter++) {
		// Record start time
		auto const start = std::chrono::high_resolution_clock::now();

		cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
		bodyForce << <n_blocks, block_size >> >(d_p.pos, d_p.vel, n_bodies);
		cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

		for (auto i = 0; i < n_bodies; i++) { // integrate position
			p.pos[i].x += p.vel[i].x*t_step;
			p.pos[i].y += p.vel[i].y*t_step;
			p.pos[i].z += p.vel[i].z*t_step;
		}

		auto const finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;

		if (iter > 1) { // First iter is warm up
			totalTime += elapsed;
		}
		printf("Iteration %d: %f seconds\n", iter, elapsed.count());
	}
	std::cout << "Total Run time for iterations 2 through " << n_iters << ": " << totalTime.count() << " seconds" << std::endl;
	std::cout << "Number of bodies: " << n_bodies << std::endl;
	std::cout << "Number of iterations per second is: " << (n_iters - 1) / totalTime.count() << std::endl;
}