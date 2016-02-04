#ifndef __CUDA_RENDERER__
#define __CUDA_RENDERER__

#include "..\Others\PerseusDefines.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "CUDADefines.h"

__host__ void initialiseRenderer(int width, int height);
__host__ void shutdownRenderer();

__host__ void renderObjectCUDA_SO_EF(float4 *vertices, int faceCount, int objectId, float *h_pmMatrix, int *h_viewTransform, int widthRender, int heightRender);

__global__ void renderScene_SO_global(unsigned char *fill, unsigned char *objects, unsigned int *zbuffer, unsigned int *zbufferInverse,
									  int4 *rois, int faceCount, int objectId, int widthRender, int heightRender);

#endif