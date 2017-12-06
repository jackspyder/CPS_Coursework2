#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <ostream>
#include <iostream>

#define SOFTENING 1e-9f //softening factor
#define G 0.6 //define gravitational constant
#define dt 0.2 //Define timestep


//Define Body with properties
typedef struct
{
	float x, y, z, vx, vy, vz, m;
} Body;

float randFloat()
{
	return 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

}

//Randomise initial body properties.
void randomizeBodies(float *data, int n) {
	for (int i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		
	}
}


int main(const int argc, const char** argv) {

	int nBodies = 4;
	if (argc > 1) nBodies = atoi(argv[1]);
	int bytes = nBodies * sizeof(Body);
	float *buf = (float*)malloc(bytes);
	Body *p = (Body*)buf;

	randomizeBodies(buf, 6 * nBodies); // generate Init pos / vel data
	for (int i = 0; i<nBodies; i++)
	{
		std::cout << "body number: " << i << std::endl;
		std::cout << p[i].x << std::endl;
		std::cout << p[i].y << std::endl;
		std::cout << p[i].z << std::endl;
		std::cout << p[i].vx << std::endl;
		std::cout << p[i].vy << std::endl;
		std::cout << p[i].vz << std::endl;

	}
	
}