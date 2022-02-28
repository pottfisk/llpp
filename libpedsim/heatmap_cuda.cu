#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ped_model.h"

#include <stdio.h>


__global__ void increase_heat(Ped::Tagent* agents){
    int id = blockIdx.x*blockDim.x +threadIdx.x;
    agents[id].getDesiredX();
    agents[id].getDesiredY();
}

void intensify_heatmap(Ped::Tagent* agents){
    Ped::Tagent* a_device;
    cudaMalloc(&a_device, sizeof(agents));
	cudaMemcpy(a_device, &agents, sizeof(agents), cudaMemcpyHostToDevice);
	increase_heat<<<1, agents.size()>>>(a_device);

}

int main(void){}
