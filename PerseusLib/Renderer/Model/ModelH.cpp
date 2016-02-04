#include "ModelH.h"

#include "..\..\CUDA\CUDAEngine.h"

using namespace Renderer::Model3D;

ModelH::ModelH(void)
{
	isAllocated = false;
}

ModelH::~ModelH(void)
{
	if (isAllocated)
	{
		delete verticesVector; 
		delete verticesVectorPreP;
		delete verticesGPUBuff;
	
		//TODO if shutdown optimiser this will be deallocated. find some way to check allocation. (maybe)
		//perseusSafeCall(cudaFree(verticesGPU));
	}
}