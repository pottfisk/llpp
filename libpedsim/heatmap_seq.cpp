// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// Sets up the heatmap
void Ped::Model::setupHeatmapSeq()
{
        int *hm; 
	int *bhm;
	int *shm;
	cout << "In setup \n";
	cudaMalloc(&hm,SIZE*SIZE*sizeof(int));
	cudaMemset(&hm,0,SIZE*SIZE*sizeof(int));
    cudaMalloc(&shm,SCALED_SIZE*SCALED_SIZE*sizeof(int));
	cudaMalloc(&bhm, SCALED_SIZE*SCALED_SIZE*sizeof(int));
	cudaMalloc(&list_device, SIZE*SIZE*sizeof(int));
	heatmap = (int**)malloc(SIZE*sizeof(int*));
	cudaMalloc(&heatmap_device, SIZE*sizeof(int *));
	cudaMalloc(&scaled_heatmap_device, SCALED_SIZE * SCALED_SIZE * sizeof(int));
	cudaMalloc(&blurred_heatmap_device, SCALED_SIZE * SCALED_SIZE * sizeof(int));

	blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
	blurred_heatmap_linear = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	
	device_setup_host(hm,shm);

	cout << "In setup 2\n";	
	//cudaMemcpy(heatmap_device, heatmap, sizeof(heatmap), cudaMemcpyHostToDevice);

	
	// for (int i = 0; i < SCALED_SIZE; i++)
	// {
	//   //	scaled_heatmap[i] = shm + SCALED_SIZE*i;
	// 	//blurred_heatmap[i] = bhm + SCALED_SIZE*i;
		
	// }
	//	cudaMemcpy(scaled_heatmap_device, scaled_heatmap, sizeof(scaled_heatmap), cudaMemcpyHostToDevice);


}



// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapSeq()
{

  int list[SIZE*SIZE] = {};
	// for (int x = 0; x < SIZE; x++)
	// {
	// 	for (int y = 0; y < SIZE; y++)
	// 	{shm + scaledshm + scaled
	// 		// heat fades
	// 		heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
	// 	}
	// }
	// Count how many agents want to go to each location
	for (int i = 0; i < agents.size(); i++)
	{
		Ped::Tagent* agent = agents[i];
		int x = agent->getDesiredX();
		int y = agent->getDesiredY();

		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
		{
			continue;
		}
		list[y*SIZE + x] += 1;
	}
	cudaMemcpy(list_device, list, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
	Ped::Model::scale_heatmap();
	//cudaMemcpy((void *)blurred_heatmap_linear, (void *)blurred_heatmap_device, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < SCALED_SIZE; i++){
	//	blurred_heatmap[i] = blurred_heatmap_linear + i*SCALED_SIZE;
	//}


	// //Scale the data for visual representation
	// for (int y = 0; y < SCALED_SIZE; y++)
	// {
	// 	for (int x = 0; x < SCALED_SIZE; x++)
	// 	{	
	// 		//if(blurred_heatmap[y][x] != 0){
	// 		cout << "BLURRED HOST: " << blurred_heatmap[y][x] << "\n";
	// 		//}
	// 	}
	// }

	// Weights for blur filter
	//const int w[5][5] = {
	//	{ 1, 4, 7, 4, 1 },
	//	{ 4, 16, 26, 16, 4 },
	//	{ 7, 26, 41, 26, 7 },
	//	{ 4, 16, 26, 16, 4 },
	//	{ 1, 4, 7, 4, 1 }
	//};

// #define WEIGHTSUM 273
// 	// Apply gaussian blurfilter		       
// 	for (int i = 2; i < SCALED_SIZE - 2; i++)
// 	{
// 		for (int j = 2; j < SCALED_SIZE - 2; j++)
// 		{
// 			int sum = 0;
// 			for (int k = -2; k < 3; k++)
// 			{
// 				for (int l = -2; l < 3; l++)
// 				{
// 					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];

// 				}
// 			}
// 			int value = sum / WEIGHTSUM;
// 			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
// 		}
// 	}
}

int Ped::Model::getHeatmapSize() const {
	return SCALED_SIZE;
}
