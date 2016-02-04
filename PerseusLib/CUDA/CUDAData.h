#ifndef __PERSEUS_CUDA_GPU_DATA__
#define __PERSEUS_CUDA_GPU_DATA__

struct CUDAData
{
	int widthFull, heightFull;
	int widthROI, heightROI;
	int bufferSize;

	int viewCount, objectCount;

	cudaArray *arrayScharr;

	int *dtVImage;
	float *dtZImage;
	int *dtImagePosYT1;
	float *dtImageT1;

	float *hKernelConvolution;

	cudaArray *arrayHeaviside;

	int histogramSize;
	float dpose[7];
	float2 *histograms;
	float3 *dfxTranslation, *dfxResultTranslation;
	float4 *dfxRotation, *dfxResultRotation;

	cudaChannelFormatDesc descRendererVertices;

	unsigned char *fill;
	unsigned char *depthMapObject;
	unsigned char *objects;

	unsigned int *zbuffer;
	unsigned int *zbufferInverse;
	unsigned int *depthMap;
	unsigned int *depthMapInverse;
	
	int roiGenerated[6];

	int4 *d_rois, *h_rois;
	
	int roisSize;
};

#endif