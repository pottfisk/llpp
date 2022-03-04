#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ped_model.h"

#include <stdio.h>
__global__ void device_setup(int** heatmap_device, int* hm, int** scaled_heatmap_device, int* shm,int scaled_size, int size){
	for (int i = 0; i < size; i++)
	{
		heatmap_device[i] = hm + size*i;
		//cout << heatmap[i] << "\n";
	}
	for (int i = 0; i < scaled_size; i++)
	{
	    scaled_heatmap_device[i] = shm + scaled_size*i;
		//blurred_heatmap[i] = bhm + scaled_size*i;
	}
    printf("Device setup done\n");
}

__global__ void scale_on_device(int** heatmap, int** scaled_heatmap, int* blurred_heatmap, int* list, int size,int cellsize,int scaled_size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int blockY = blockIdx.y*blockDim.y;
    
    __shared__ int shared_heatmap[16][16];
    shared_heatmap[threadIdx.y][threadIdx.x] =  heatmap[idy][idx];

    shared_heatmap[threadIdx.y][threadIdx.x] = (int)round(shared_heatmap[threadIdx.y][threadIdx.x] * 0.80);
    shared_heatmap[threadIdx.y][threadIdx.x] += list[idy*size + idx]*40;

    shared_heatmap[threadIdx.y][threadIdx.x] = shared_heatmap[threadIdx.y][threadIdx.x] < 255 ? shared_heatmap[threadIdx.y][threadIdx.x] : 255;
    int value = shared_heatmap[threadIdx.y][threadIdx.x];
    __syncthreads();
    heatmap[idy][idx] = shared_heatmap[threadIdx.y][threadIdx.x];
    for (int cellY = 0; cellY < cellsize; cellY++)
    {
    	for (int cellX = 0; cellX < cellsize; cellX++)
    	{   
    		scaled_heatmap[idy * cellsize + cellY][idx * cellsize + cellX] = value;
    		//printf("Scale device %d \n", 2);
    	}
    }


    const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

    #define WEIGHTSUM 273
	// Apply gaussian blurfilter		       
            if(idy > 1 && idy < scaled_size - 2 && idx > 1 && idx < scaled_size-2){
                int sum = 0;
                for (int k = -2; k < 3; k++)
                {
                    for (int l = -2; l < 3; l++)
                    {
                        sum += w[2 + k][2 + l] * scaled_heatmap[idy + k][idx + l];			
                    }
                }
                int value = sum / WEIGHTSUM;
                //rintf("value blur:  %d\n", value);
                blurred_heatmap[idy*scaled_size + idx] = 0x00FF0000 | value << 24;
            }


}

void Ped::Model::scale_heatmap(){
    dim3 dimBlock(SIZE/64, SIZE/64);
    dim3 dimGrid(64, 64);
    //printf("In scale\n");
    scale_on_device<<<dimGrid, dimBlock>>>(heatmap_device,scaled_heatmap_device,blurred_heatmap_device, list_device,SIZE,CELLSIZE,SCALED_SIZE);
    
}

void Ped::Model::device_setup_host(int *hm, int *shm){
device_setup<<<1,1>>>(heatmap_device, hm,scaled_heatmap_device,shm,SCALED_SIZE,SIZE);
}

int main(void){}
