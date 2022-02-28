#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ped_model.h"

#include <stdio.h>


__global__ void scale_on_device(int** heatmap, int size){
    int idx = blockIdx.x*blockDim.x +threadIdx.x;
    int idy = blockIdx.y*blockDim.y +threadIdx.y;
    printf("Num: %d %d", idx,idy);
}

void Ped::Model::scale_heatmap(int** a_device){
    dim3 dimBlock(1, 1);
    dim3 dimGrid(SIZE/dimBlock.y, SIZE/dimBlock.x);

//    Mat<<<dimGrid, dimBlock >>>(d_a, d_b, d_c, n);


	scale_on_device<<<dimGrid, dimBlock>>>(a_device,SIZE);

}

int main(void){}
