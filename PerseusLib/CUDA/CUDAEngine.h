#ifndef __CUDA_ENGINE_DEVICE__
#define __CUDA_ENGINE_DEVICE__

#include "..\Others\PerseusDefines.h"
#include "..\Objects\Object3D.h"
#include "..\Objects\View3D.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace Perseus::Objects;

#include "CUDADefines.h"

void initialiseCUDA(int width, int height, float* heavisideFunction, int heavisideFunctionSize);
void shutdownCUDA();

void registerObjectImage(Object3D* object, View3D* view, bool renderingFromGPU);
void registerObjectAndViewGeometricData(Object3D* object, View3D* view);

void processDTSihluetteLSDXDY(Object3D* object, View3D* view, int bandSize);
void processAndGetEFFirstDerivatives(Object3D* object, View3D* view);

void getProcessedDataDTSihluetteLSDXDY(Object3D* object, View3D* view);
void getProcessedDataEFFirstDerivatives(Object3D* object, View3D* view);
void getProcessedDataRendering(Object3D* object, View3D* view);

void renderObjectCUDA_SO(Object3D *object, View3D *view);

#endif