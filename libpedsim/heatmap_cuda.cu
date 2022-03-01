#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ped_model.h"

#include <stdio.h>
__global__ void device_setup(int** heatmap_device, int* hm,int** scaled_heatmap_device,int* shm,int scaled_size, int size){
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
printf("here2222");
}

__global__ void scale_on_device(int** heatmap, int** scaled_heatmap, int* list, int size,int cellsize){
    int idx = blockIdx.x*blockDim.x +threadIdx.x;
    int idy = blockIdx.y*blockDim.y +threadIdx.y;

  //  heatmap[idy][idx] = (int)round(heatmap[idy][idx] * 0.80);
   // heatmap[idy][idx] += list[idy*size + idx]*40;

//    heatmap[idy][idx] = heatmap[idy][idx] < 255 ? heatmap[idy][idx] : 255;
    //int value = heatmap[idy][idx];
    printf("value %d \n",2);
    // for (int cellY = 0; cellY < cellsize; cellY++)
    // {
    // 	for (int cellX = 0; cellX < cellsize; cellX++)
    // 	{
    // //		scaled_heatmap[idy * cellsize + cellY][idx * cellsize + cellX] = value;
    // 		//printf("Scale device %d \n", 2);
    // 	}
    // }

}

void Ped::Model::scale_heatmap(){
    dim3 dimBlock(SIZE/64, SIZE/64);
    dim3 dimGrid(64, 64);
    printf("In scale\n");
    scale_on_device<<<dimGrid, dimBlock>>>(heatmap_device,scaled_heatmap_device,list_device,SIZE,CELLSIZE);
    
}

void Ped::Model::device_setup_host(int *hm, int *shm){
device_setup<<<1,1>>>(heatmap_device, hm,scaled_heatmap_device,shm,SCALED_SIZE,SIZE);
}

int main(void){}
