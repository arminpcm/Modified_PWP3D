#ifndef __VIPRAD_IMAGE__
#define __VIPRAD_IMAGE__

#include "..\Utils\FileUtils.h"
#include "..\Others\PerseusDefines.h"
#include <stdlib.h>
#include <string.h>

#ifndef PIXEL_UCHAR
#define PIXEL_UCHAR unsigned char
#endif

#ifndef PIXEL_INT
#define PIXEL_INT int
#endif

#ifndef PIXEL_UINT
#define PIXEL_UINT unsigned int
#endif

#ifndef PIXEL_FLOAT
#define PIXEL_FLOAT float
#endif

#include <cuda_runtime_api.h>

namespace Perseus
{
	namespace Primitives
	{
		class PerseusImage
		{
		public:
			size_t totalSize;

			void* imageData;
			void* imageDataGPU;

			PIXEL_FLOAT *pixelsF;
			PIXEL_FLOAT *pixelsFGPU;

			PIXEL_INT *pixelsI;
			PIXEL_INT *pixelsIGPU;

			PIXEL_UCHAR* pixelsGPU;
			PIXEL_UINT* zbufferGPU;
			PIXEL_UINT* zbufferInverseGPU;
			PIXEL_UCHAR* objectsGPU;

			PIXEL_UCHAR* pixels;
			PIXEL_UINT* zbuffer;
			PIXEL_UINT* zbufferInverse;
			PIXEL_UCHAR* objects;

			PIXEL_UINT* original;
			PIXEL_UINT* originalInverse;

			bool hasZBuffer;
			bool useCudaAlloc;

			int width, height, bpp;

			enum ImageType { IMAGE_RGBA, IMAGE_GRAY, IMAGE_FLOAT, IMAGE_INT }imageType;

			PerseusImage(int width, int height, ImageType imageType, bool clear, bool hasZBuffer, bool useCudaAlloc)
			{
				this->width = width;
				this->height = height;

				this->imageType = imageType;
				this->useCudaAlloc = useCudaAlloc; 
				this->hasZBuffer = hasZBuffer;

				switch (imageType)
				{
				case IMAGE_RGBA: 
					{
						bpp = 4; 
						if (useCudaAlloc)
						{
							totalSize = sizeof(PIXEL_UCHAR) * width * height * bpp;
							cudaMallocHost(&imageData, totalSize); 
							cudaMalloc((void**)&imageDataGPU, totalSize);

							pixels = (PIXEL_UCHAR*) imageData;
							pixelsGPU = (PIXEL_UCHAR*) imageDataGPU;
						}
						else { pixels = new PIXEL_UCHAR[width * height * bpp]; }
					}
					break;
				case IMAGE_GRAY: 
					{
						bpp = 1; 
						size_t sizeObjects = sizeof(PIXEL_UCHAR) * width * height;
						size_t sizeImageZBuffer = sizeof(PIXEL_UINT) * width * height;
						size_t sizeImageZBufferInverse = sizeof(PIXEL_UINT) * width * height;
						size_t sizeImagePixels = sizeof(PIXEL_UCHAR) * width * height * bpp;

						if (useCudaAlloc)
						{
							totalSize = sizeObjects + sizeImageZBuffer + sizeImageZBufferInverse + sizeImagePixels;

							cudaMallocHost(&imageData, totalSize);
							cudaMalloc((void**)&imageDataGPU, totalSize);

							objects = (PIXEL_UCHAR*) imageData;
							zbuffer = (PIXEL_UINT*) ((PIXEL_UCHAR*)objects +  sizeImagePixels);
							zbufferInverse = (PIXEL_UINT*) ((PIXEL_UCHAR*)zbuffer + sizeImageZBuffer);
							pixels = (PIXEL_UCHAR*) ((PIXEL_UCHAR*)zbufferInverse + sizeImageZBufferInverse);

							objectsGPU = (PIXEL_UCHAR*) imageDataGPU;
							zbufferGPU = (PIXEL_UINT*) ((PIXEL_UCHAR*)objectsGPU +  sizeImagePixels);
							zbufferInverseGPU = (PIXEL_UINT*) ((PIXEL_UCHAR*)zbufferGPU + sizeImageZBuffer);
							pixelsGPU = (PIXEL_UCHAR*) ((PIXEL_UCHAR*)zbufferInverseGPU + sizeImageZBufferInverse);
						}
						else
						{
							pixels = new PIXEL_UCHAR[width * height * bpp];
							zbuffer = new PIXEL_UINT[width * height];
							zbufferInverse = new PIXEL_UINT[width * height];
							objects = new PIXEL_UCHAR[width * height];
						}
					}
					break;
				case IMAGE_FLOAT:
					{
						bpp = 1;
						if (useCudaAlloc)
						{
							totalSize = sizeof(PIXEL_FLOAT) * width * height * bpp;
							cudaMallocHost(&imageData, totalSize); 
							cudaMalloc((void**)&imageDataGPU, totalSize);

							pixelsF = (PIXEL_FLOAT*) imageData;
							pixelsFGPU = (PIXEL_FLOAT*) imageDataGPU;
						}
						else { pixelsF = new PIXEL_FLOAT[width * height * bpp]; }
					}
					break;
				case IMAGE_INT:
					{
						bpp = 1;
						if (useCudaAlloc)
						{
							totalSize = sizeof(PIXEL_INT) * width * height * bpp;
							cudaMallocHost(&imageData, totalSize); 
							cudaMalloc((void**)&imageDataGPU, totalSize);

							pixelsI = (PIXEL_INT*) imageData;
							pixelsIGPU = (PIXEL_INT*) imageDataGPU;
						}
						else { pixelsI = new PIXEL_INT[width * height * bpp]; }
					}
					break;
				}

				if (clear)
				{
					this->Clear(0);
					this->ClearZBuffer();
				}
			}

			PerseusImage() { }

			~PerseusImage()
			{
				if (useCudaAlloc)
				{
					cudaFreeHost(imageData);
					cudaFree(imageDataGPU);
				}
				else
				{
					switch (imageType)
					{
					case IMAGE_RGBA: delete pixels; break;
					case IMAGE_GRAY: delete pixels; break;
					case IMAGE_FLOAT: delete pixelsF; break;
					case IMAGE_INT: delete pixelsI; break;
					}

					if (hasZBuffer)
					{
						delete zbuffer;
						delete zbufferInverse;
						delete objects;
					}
				}
			}

			void Clear(VBYTE color) 
			{ 
				switch (imageType)
				{
				case IMAGE_RGBA: memset(pixels, color, sizeof(PIXEL_UCHAR) * width * height * bpp); break;
				case IMAGE_GRAY: memset(pixels, color, sizeof(PIXEL_UCHAR) * width * height * bpp); break;
				case IMAGE_FLOAT: memset(pixelsF, color, sizeof(PIXEL_FLOAT) * width * height * bpp); break;
				case IMAGE_INT: memset(pixelsI, color, sizeof(PIXEL_INT) * width * height * bpp); break;
				}
			}
			void ClearF(float color) 
			{ 
				for (int i=0; i < width * height; i++) pixelsF[i] = color;
			}
			void ClearZBuffer()
			{
				if (this->hasZBuffer)
				{
					memset(zbuffer, int(MAX_INT), sizeof(PIXEL_UINT) * width * height);
					memset(zbufferInverse, 0, sizeof(PIXEL_UINT) * width * height);
					memset(objects, 0, sizeof(PIXEL_UCHAR) * width * height);
				}
			}

			void UpdateGPUFromCPU() { if (useCudaAlloc) cudaMemcpy(imageDataGPU, imageData, totalSize, cudaMemcpyHostToDevice); }
			void UpdateCPUFromGPU() { if (useCudaAlloc) cudaMemcpy(imageData, imageDataGPU, totalSize, cudaMemcpyDeviceToHost); }

			void LoadFromPixels(unsigned char* pixels)
			{
				memcpy(this->pixels, pixels, this->width * this->height * sizeof(unsigned char) * 4);
			}
		};
	}
}

#endif