#include "HistogramEngine.h"
#include "ImageUtils.h"

#include <omp.h>

using namespace Perseus::Utils;

HistogramEngine* HistogramEngine::instance;

HistogramEngine::HistogramEngine(void) { }

HistogramEngine::~HistogramEngine(void) { delete instance; }

void HistogramEngine::NormaliseHistogramVarBin(HistogramVarBin *histogram)
{
	int i, j, k, histIndex, histOffset, histNo;
	float sumHistogramForeground, sumHistogramBackground;

	sumHistogramForeground = histogram->totalForegroundPixels; 
	sumHistogramBackground = histogram->totalBackgroundPixels;

	sumHistogramForeground = (sumHistogramForeground != 0) ? 1.0f / sumHistogramForeground : 0;
	sumHistogramBackground = (sumHistogramBackground != 0) ? 1.0f / sumHistogramBackground : 0;

	for (histNo = 0; histNo<histogram->noHistograms; histNo++)
	{
		histOffset = histogram->histOffsets[histNo];

		for (i=0; i<histogram->noBins[histNo]; i++) for (j=0; j<histogram->noBins[histNo]; j++) for (k=0; k<histogram->noBins[histNo]; k++)
		{
			histIndex = (i + j * histogram->noBins[histNo]) * histogram->noBins[histNo] + k;
			if (histogram->alreadyInitialised)
			{
				histogram->normalised[histOffset + histIndex].x = 
					histogram->normalised[histOffset + histIndex].x * (1.0f - histogram->mergeAlphaForeground) + 
					histogram->notnormalised[histOffset + histIndex].x * sumHistogramForeground * histogram->mergeAlphaForeground;

				histogram->normalised[histOffset + histIndex].y = 
					histogram->normalised[histOffset + histIndex].y * (1.0f - histogram->mergeAlphaBackground) + 
					histogram->notnormalised[histOffset + histIndex].y * sumHistogramBackground * histogram->mergeAlphaBackground;
			}
			else
			{
				histogram->normalised[histOffset + histIndex].x = histogram->notnormalised[histOffset + histIndex].x * sumHistogramForeground;
				histogram->normalised[histOffset + histIndex].y = histogram->notnormalised[histOffset + histIndex].y * sumHistogramBackground;
			}
		}
	}

	if (!histogram->alreadyInitialised) histogram->alreadyInitialised = true;
}


void HistogramEngine::BuildHistogramVarBin(HistogramVarBin *histogram, PerseusImage *mask, PerseusImage* image, int objectId)
{
	int i, j, idx, r, g, b;

	idx = 0;
	for (j = 0; j < image->height; j++) for (i = 0; i < image->width; i++)
	{
		idx = i + j * mask->width;
		
		r = image->pixels[idx * image->bpp + 0]; g = image->pixels[idx * image->bpp + 1]; b = image->pixels[idx * image->bpp + 2];

		if (mask->objects[idx] != 0 && (mask->objects[idx] - 1) == objectId) 
			histogram->AddPoint(1, 0, r, g, b, i, j);
		else histogram->AddPoint(0, 1, r, g, b, i, j);
	}
}


void HistogramEngine::BuildHistogramVarBin(HistogramVarBin *histogram, bool *mask, PerseusImage* image, int objectId)
{
	int i, j, idx, r, g, b;

	idx = 0;
	for (j = 0; j < image->height; j++) for (i = 0; i < image->width; i++)
	{
		idx = i + j * image->width;
		
		r = image->pixels[idx * image->bpp + 0]; g = image->pixels[idx * image->bpp + 1]; b = image->pixels[idx * image->bpp + 2];

		if (mask[idx]) histogram->AddPoint(1, 0, r, g, b, i, j);
		else histogram->AddPoint(0, 1, r, g, b, i, j);
	}
}

void HistogramEngine::BuildHistogramVarBin(HistogramVarBin *histogram, PerseusImage *mask, float *dt, PerseusImage* image, int objectId)
{
	int i, j, idx, r, g, b;

	idx = 0;
	for (j = 0; j < image->height; j++) for (i = 0; i < image->width; i++)
	{
		idx = i + j * mask->width;

		if (dt[idx] == 0)
		{
			r = image->pixels[idx * image->bpp + 0]; g = image->pixels[idx * image->bpp + 1]; b = image->pixels[idx * image->bpp + 2];

			if (mask->objects[idx] != 0 && (mask->objects[idx] - 1) == objectId) 
				histogram->AddPoint(1, 0, r, g, b, i, j);
			else histogram->AddPoint(0, 1, r, g, b, i, j);
		}
	}
}

void HistogramEngine::UpdateVarBinHistogram(PerseusImage* originalImage, PerseusImage* mask, HistogramVarBin *histogram, int objectId)
{
	if (originalImage->imageType != PerseusImage::IMAGE_RGBA)
	{
		printf("Unsupported image type for histogram!\n");
		exit(1);
	}
	
	this->BuildHistogramVarBin(histogram, mask, originalImage, objectId);
	this->NormaliseHistogramVarBin(histogram);
}

void HistogramEngine::UpdateVarBinHistogram(PerseusImage* originalImage, PerseusImage* mask, float* dt, HistogramVarBin *histogram, int objectId)
{
	if (originalImage->imageType != PerseusImage::IMAGE_RGBA)
	{
		printf("Unsupported image type for histogram!\n");
		exit(1);
	}
	
	this->BuildHistogramVarBin(histogram, mask, dt, originalImage, objectId);
	this->NormaliseHistogramVarBin(histogram);
}

void HistogramEngine::UpdateVarBinHistogram(Object3D* object, View3D* view, PerseusImage* originalImage, PerseusImage* mask)
{
	if (originalImage->imageType != PerseusImage::IMAGE_RGBA)
	{
		printf("Unsupported image type for histogram!\n");
		exit(1);
	}

	ImageUtils::Instance()->Copy(mask, object->imageHistogramMask[view->viewId]);
	this->UpdateVarBinHistogram(originalImage, mask, object->histogramVarBin[view->viewId], object->objectId);
	object->histogramVarBin[view->viewId]->UpdateGPUFromCPU();
}

void HistogramEngine::UpdateVarBinHistogram(Object3D* object, View3D* view, PerseusImage* originalImage, bool* mask)
{
	if (originalImage->imageType != PerseusImage::IMAGE_RGBA)
	{
		printf("Unsupported image type for histogram!\n");
		exit(1);
	}

	this->BuildHistogramVarBin(object->histogramVarBin[view->viewId], mask, originalImage, object->objectId);
	this->NormaliseHistogramVarBin(object->histogramVarBin[view->viewId]);
	object->histogramVarBin[view->viewId]->UpdateGPUFromCPU();
}

void HistogramEngine::UpdateVarBinHistogram(Object3D* object, View3D* view, PerseusImage* originalImage, Pose3D* pose)
{
	if (originalImage->imageType != PerseusImage::IMAGE_RGBA)
	{
		printf("Unsupported image type for histogram!\n");
		exit(1);
	}

	DrawingEngine::Instance()->Draw(object, view, pose, object->imageHistogramMask[view->viewId], DrawingEngine::RENDERING_FILL);
	this->UpdateVarBinHistogram(object, view, originalImage, object->imageHistogramMask[view->viewId]);
	object->histogramVarBin[view->viewId]->UpdateGPUFromCPU();
}

void HistogramEngine::SetVarBinHistogram(Object3D* object, View3D* view, float2 *normalised)
{
	memcpy(object->histogramVarBin[view->viewId]->normalised, normalised, sizeof(float2) * object->histogramVarBin[view->viewId]->fullHistSize);
	object->histogramVarBin[view->viewId]->UpdateGPUFromCPU();
}