#include "CUDAEngine.h"

#include "CUDAData.h"
#include "CUDADT.h"
#include "CUDAConvolution.h"
#include "CUDAScharr.h"
#include "CUDAEF.h"
#include "CUDARenderer.h"

CUDAData *cudaData;

void initialiseCUDA(int width, int height, float* heavisideFunction, int heavisideFunctionSize)
{
	cudaData = new CUDAData();

	initialiseRenderer(width, height);
	initialiseScharr(width, height);
	initialiseDT(width, height);
	initialiseConvolution(width, height);
	initialiseEF(width, height, heavisideFunction, heavisideFunctionSize);
}

void shutdownCUDA()
{
	shutdownRenderer();
	shutdownScharr();
	shutdownDT();
	shutdownConvolution();
	shutdownEF();

	cudaThreadExit();
}

void registerObjectImage(Object3D* object, View3D* view, bool renderingFromGPU)
{
	int viewId = view->viewId;

	int *roiGenerated = object->roiGenerated[viewId];
	int *roiNormalised = object->roiNormalised[viewId];

	unsigned char *objects;
	unsigned int *zbuffer, *zbufferInverse; 
	cudaMemcpyKind renderingSource;
	if (renderingFromGPU) 
	{
		objects = cudaData->objects;
		zbuffer = cudaData->zbuffer;
		zbufferInverse = cudaData->zbufferInverse;
		renderingSource = cudaMemcpyDeviceToDevice;
	}
	else
	{
		objects = object->imageFill[viewId]->objects;
		zbuffer = object->imageFill[viewId]->zbuffer;
		zbufferInverse = object->imageFill[viewId]->zbufferInverse;
		renderingSource = cudaMemcpyHostToDevice;
	}

	unsigned char *objectsGPUROI = object->imageFill[viewId]->objectsGPU;
	unsigned int *zbufferGPUROI = object->imageFill[viewId]->zbufferGPU;
	unsigned int *zbufferInverseGPUROI = object->imageFill[viewId]->zbufferInverseGPU;

	uchar4 *cameraGPU = (uchar4*) view->imageRegistered->pixelsGPU;
	uchar4 *cameraGPUROI = (uchar4*) object->imageCamera[viewId]->pixelsGPU;

	cudaData->widthFull = object->imageFill[viewId]->width;
	cudaData->heightFull = object->imageFill[viewId]->height;
	cudaData->widthROI = roiGenerated[4]; cudaData->heightROI = roiGenerated[5];

	roiNormalised[0] = 0; roiNormalised[1] = 0;
	roiNormalised[2] = roiGenerated[4]; roiNormalised[3] = roiGenerated[5]; 
	roiNormalised[4] = roiGenerated[4]; roiNormalised[5] = roiGenerated[5];

	perseusSafeCall(cudaMemcpy2D(objectsGPUROI, cudaData->widthROI, objects + roiGenerated[0] + roiGenerated[1] * cudaData->widthFull, 
		cudaData->widthFull, cudaData->widthROI, cudaData->heightROI, renderingSource));
	perseusSafeCall(cudaMemcpy2D(zbufferGPUROI, cudaData->widthROI * sizeof(uint1), zbuffer + roiGenerated[0] + roiGenerated[1] * cudaData->widthFull, 
		cudaData->widthFull * sizeof(uint1), cudaData->widthROI * sizeof(uint1), cudaData->heightROI, renderingSource));
	perseusSafeCall(cudaMemcpy2D(zbufferInverseGPUROI, cudaData->widthROI * sizeof(uint1), zbufferInverse + roiGenerated[0] + roiGenerated[1] * cudaData->widthFull, 
		cudaData->widthFull * sizeof(uint1), cudaData->widthROI * sizeof(uint1), cudaData->heightROI, renderingSource));

	perseusSafeCall(cudaMemcpy2D(cameraGPUROI, cudaData->widthROI * sizeof(uchar4), cameraGPU + roiGenerated[0] + roiGenerated[1] * cudaData->widthFull,
		cudaData->widthFull * sizeof(uchar4), cudaData->widthROI * sizeof(uchar4), cudaData->heightROI, cudaMemcpyDeviceToDevice));
}

void registerObjectAndViewGeometricData(Object3D* object, View3D* view)
{
	float rotationParameters[7];

	object->pose->rotation->Get(rotationParameters);

	registerObjectGeometricData(rotationParameters, object->invPMMatrix[view->viewId]);
	registerViewGeometricData(view->renderView->invP, view->renderView->projectionParams.all, view->renderView->view);
}

void processDTSihluetteLSDXDY(Object3D* object, View3D* view, int bandSize)
{
	int viewId = view->viewId;

	int *roi = object->roiNormalised[viewId];

	unsigned char *objectsGPUROI = object->imageFill[viewId]->objectsGPU;
	unsigned char *sihluetteGPUROI = object->imageSihluette[viewId]->pixelsGPU;

	float *dtGPUROI = object->dt[viewId]->pixelsFGPU;
	int *dtPosXGPUROI = object->dtPosX[viewId]->pixelsIGPU;
	int *dtPosYGPUROI = object->dtPosY[viewId]->pixelsIGPU;
	float *dtDXGPUROI = object->dtDX[viewId]->pixelsFGPU;
	float *dtDYGPUROI = object->dtDY[viewId]->pixelsFGPU;

	computeSihluette(objectsGPUROI, sihluetteGPUROI, roi[4], roi[5], 1.0f);
	processDT(dtGPUROI, dtPosXGPUROI, dtPosYGPUROI, sihluetteGPUROI, objectsGPUROI, roi, bandSize);
	computeDerivativeXY(dtGPUROI, dtDXGPUROI, dtDYGPUROI, roi[4], roi[5]);
}

void processAndGetEFFirstDerivatives(Object3D* object, View3D* view)
{
	int viewId = view->viewId;

	float dpose[7];

	int *roiNormalised = object->roiNormalised[viewId];
	int *roiGenerated = object->roiGenerated[viewId];

	float2 *histogram = (float2*) object->histogramVarBin[viewId]->normalisedGPU;

	uchar4 *cameraGPUROI = (uchar4*) object->imageCamera[viewId]->pixelsGPU;

	unsigned char *objectsGPUROI = object->imageFill[viewId]->objectsGPU;
	unsigned int *zbufferGPUROI = object->imageFill[viewId]->zbufferGPU;
	unsigned int *zbufferInverseGPUROI = object->imageFill[viewId]->zbufferInverseGPU;

	float *dtGPUROI = object->dt[viewId]->pixelsFGPU;
	int *dtPosXGPUROI = object->dtPosX[viewId]->pixelsIGPU;
	int *dtPosYGPUROI = object->dtPosY[viewId]->pixelsIGPU;

	float *dtDXGPUROI = object->dtDX[viewId]->pixelsFGPU;
	float *dtDYGPUROI = object->dtDY[viewId]->pixelsFGPU;

	processEFD1(dpose, roiNormalised, roiGenerated, histogram, cameraGPUROI, objectsGPUROI, zbufferGPUROI, zbufferInverseGPUROI, dtGPUROI, dtPosXGPUROI,
		dtPosYGPUROI, dtDXGPUROI, dtDYGPUROI);

	object->dpose->Set(dpose);
}

void getProcessedDataDTSihluetteLSDXDY(Object3D* object, View3D* view)
{
	int viewId = view->viewId;

	int *roi = object->roiGenerated[viewId];

	float *dtGPUROI = object->dt[viewId]->pixelsFGPU;
	int *dtPosXGPUROI = object->dtPosX[viewId]->pixelsIGPU;
	int *dtPosYGPUROI = object->dtPosY[viewId]->pixelsIGPU;
	unsigned char* sihluetteGPUROI = object->imageSihluette[viewId]->pixelsGPU;
	float *dtDXGPUROI = object->dtDX[viewId]->pixelsFGPU;
	float *dtDYGPUROI = object->dtDY[viewId]->pixelsFGPU;

	float *dt = object->dt[viewId]->pixelsF;
	int *dtPosX = object->dtPosX[viewId]->pixelsI;
	int *dtPosY = object->dtPosY[viewId]->pixelsI;
	unsigned char* sihluette = object->imageSihluette[viewId]->pixels;
	float *dtDX = object->dtDX[viewId]->pixelsF;
	float *dtDY = object->dtDY[viewId]->pixelsF;

	perseusSafeCall(cudaThreadSynchronize());

	memset(dt, 0, cudaData->widthFull * cudaData->heightFull * sizeof(float));
	memset(dtPosX, -1, cudaData->widthFull * cudaData->heightFull * sizeof(int));
	memset(dtPosY, -1, cudaData->widthFull * cudaData->heightFull * sizeof(int));
	memset(sihluette, 0, cudaData->widthFull * cudaData->heightFull * sizeof(unsigned char));
	memset(dtDX, 0, cudaData->widthFull * cudaData->heightFull * sizeof(float));
	memset(dtDY, 0, cudaData->widthFull * cudaData->heightFull * sizeof(float));

	perseusSafeCall(cudaMemcpy2D(dt + roi[0] + roi[1] * cudaData->widthFull, cudaData->widthFull * sizeof(float), 
		dtGPUROI, roi[4] * sizeof(float), roi[4] * sizeof(float), roi[5], cudaMemcpyDeviceToHost));
	perseusSafeCall(cudaMemcpy2D(dtPosX + roi[0] + roi[1] * cudaData->widthFull, cudaData->widthFull * sizeof(int), 
		dtPosXGPUROI, roi[4] * sizeof(int), roi[4] * sizeof(int), roi[5], cudaMemcpyDeviceToHost));
	perseusSafeCall(cudaMemcpy2D(dtPosY + roi[0] + roi[1] * cudaData->widthFull, cudaData->widthFull * sizeof(int), 
		dtPosYGPUROI, roi[4] * sizeof(int), roi[4] * sizeof(int), roi[5], cudaMemcpyDeviceToHost));
	perseusSafeCall(cudaMemcpy2D(sihluette + roi[0] + roi[1] * cudaData->widthFull, cudaData->widthFull * sizeof(unsigned char), 
		sihluetteGPUROI, roi[4] * sizeof(unsigned char), roi[4] * sizeof(unsigned char), roi[5], cudaMemcpyDeviceToHost));
	perseusSafeCall(cudaMemcpy2D(dtDX + roi[0] + roi[1] * cudaData->widthFull, cudaData->widthFull * sizeof(float), 
		dtDXGPUROI, roi[4] * sizeof(float), roi[4] * sizeof(float), roi[5], cudaMemcpyDeviceToHost));
	perseusSafeCall(cudaMemcpy2D(dtDY + roi[0] + roi[1] * cudaData->widthFull, cudaData->widthFull * sizeof(float), 
		dtDYGPUROI, roi[4] * sizeof(float), roi[4] * sizeof(float), roi[5], cudaMemcpyDeviceToHost));
}

void getProcessedDataRendering(Object3D* object, View3D* view)
{
	int viewId = view->viewId;

	cudaData->widthFull = object->imageFill[viewId]->width;
	cudaData->heightFull = object->imageFill[viewId]->height;

	unsigned char *fill = object->imageFill[viewId]->pixels;
	unsigned char *objects = object->imageFill[viewId]->objects;
	unsigned int *zbuffer = object->imageFill[viewId]->zbuffer;
	unsigned int *zbufferInverse = object->imageFill[viewId]->zbufferInverse; 

	perseusSafeCall(cudaMemcpy(fill, cudaData->fill, sizeof(unsigned char) * cudaData->widthFull * cudaData->heightFull, cudaMemcpyDeviceToHost));
	perseusSafeCall(cudaMemcpy(objects, cudaData->objects, sizeof(unsigned char) * cudaData->widthFull * cudaData->heightFull, cudaMemcpyDeviceToHost));
	perseusSafeCall(cudaMemcpy(zbuffer, cudaData->zbuffer, sizeof(unsigned int) * cudaData->widthFull * cudaData->heightFull, cudaMemcpyDeviceToHost));
	perseusSafeCall(cudaMemcpy(zbufferInverse, cudaData->zbufferInverse, sizeof(unsigned int) * cudaData->widthFull * cudaData->heightFull, cudaMemcpyDeviceToHost));
}

void renderObjectCUDA_SO(Object3D *object, View3D *view)
{
	int viewId = view->viewId;
	Renderer3DObject* renderObject = object->renderObject;

	renderObjectCUDA_SO_EF((float4*)renderObject->drawingModel[viewId]->verticesGPU, renderObject->drawingModel[viewId]->faceCount, 
		object->objectId, object->pmMatrix[viewId], view->renderView->view, object->imageFill[viewId]->width, object->imageFill[viewId]->height);

	memcpy(object->roiGenerated[viewId], cudaData->roiGenerated, 6 * sizeof(int));
}