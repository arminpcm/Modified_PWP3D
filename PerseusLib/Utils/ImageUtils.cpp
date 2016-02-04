#include "..\Utils\ImageUtils.h"

#include <windows.h>
#include <FreeImage.h>

using namespace Perseus::Utils;
#include <FreeImage.h>

ImageUtils* ImageUtils::instance;

ImageUtils::ImageUtils(void)
{
}

ImageUtils::~ImageUtils(void)
{
}

void ImageUtils::SaveImageToFile(PerseusImage* image, char* fileName)
{
	PerseusImage* newImage = new PerseusImage(image->width, image->height, PerseusImage::IMAGE_RGBA, false, false, false);
	this->Copy(image, newImage);

	FIBITMAP *bmp = FreeImage_ConvertFromRawBits(newImage->pixels, newImage->width, newImage->height, 
		newImage->width * newImage->bpp, newImage->bpp * 8, 0, 0, 0, false);

	FreeImage_FlipVertical(bmp);

	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(fileName, 0);
	if(fif == FIF_UNKNOWN)  { fif = FreeImage_GetFIFFromFilename(fileName); }
	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)) { FreeImage_Save(fif, bmp, fileName, 0); }

	FreeImage_Unload(bmp);
	delete newImage;
}

bool ImageUtils::LoadImageFromFile(PerseusImage* image, char* fileName)
{
	bool bLoaded = false;
	int bpp;
	FIBITMAP *bmp = 0;
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(fileName);

	if (fif == FIF_UNKNOWN) { fif = FreeImage_GetFIFFromFilename(fileName); }

	if (fif != FIF_UNKNOWN && FreeImage_FIFSupportsReading(fif))
	{
		bmp = FreeImage_Load(fif, fileName, 0);
		bLoaded = true; if (bmp == NULL) bLoaded = false;
	}

	if (bLoaded)
	{
		FreeImage_FlipVertical(bmp);

		bpp = FreeImage_GetBPP(bmp);
		switch (bpp)
		{
		case 32:
			break;
		default:
			FIBITMAP *bmpTemp = FreeImage_ConvertTo32Bits(bmp);
			if (bmp != NULL) FreeImage_Unload(bmp);
			bmp = bmpTemp;
			bpp = FreeImage_GetBPP(bmp);
			break;
		}

		memcpy(image->pixels, FreeImage_GetBits(bmp),  sizeof(unsigned char) * 4 * image->width * image->height);

		FreeImage_Unload(bmp);

		return true;
	}

	return false;
}

void ImageUtils::Copy(PerseusImage* src, PerseusImage *dst, bool includeZBuffer, bool tryOnlyPointer)
{
	int i, j, idx, idx2;

	idx = 0; idx2 = 0;

	if (dst->bpp == src->bpp)
	{
		if (tryOnlyPointer)
		{
			dst->imageType = src->imageType;
			dst->pixels = src->pixels;
		}
		else
		{
			memcpy(dst->pixels, src->pixels, src->width * src->height * sizeof(PIXEL_UCHAR) * src->bpp);
			if (includeZBuffer && src->hasZBuffer)
			{
				memcpy(dst->zbuffer, src->zbuffer, src->width * src->height * sizeof(PIXEL_UINT));
				memcpy(dst->objects, src->objects, src->width * src->height * sizeof(PIXEL_UCHAR));
				memcpy(dst->zbufferInverse, src->zbufferInverse, src->width * src->height * sizeof(PIXEL_UINT));
				dst->hasZBuffer = src->hasZBuffer;
			}
		}

		return;
	}

	if (dst->bpp > src->bpp)
	{
		for (j=0; j<dst->height; j++) for (i=0; i<dst->width; i++)
		{
			dst->pixels[idx2 + 0] = src->pixels[idx];
			dst->pixels[idx2 + 1] = src->pixels[idx];
			dst->pixels[idx2 + 2] = src->pixels[idx];

			idx2 += dst->bpp; idx++;
		}
	}

	if (dst->bpp < src->bpp)
	{
		delete dst->pixels;
		dst->imageType = src->imageType;
		dst->pixels = new PIXEL_UCHAR[src->width * src->height * src->bpp];
		dst->bpp = src->bpp;

		printf("Not implemented copy");
		exit(1);
	}
}

void ImageUtils::Overlay(PerseusImage* srcGrey, PerseusImage *destRGB, int destB, int destG, int destR)
{
	int idx, idxB;
	for (idx = 0, idxB = 0; idx < srcGrey->width * srcGrey->height; idx++, idxB += destRGB->bpp)
	{
		if (srcGrey->pixels[idx] > 0)
		{
			destRGB->pixels[idxB + 0] = destR;
			destRGB->pixels[idxB + 1] = destG;
			destRGB->pixels[idxB + 2] = destB;
		}
	}
}

void ImageUtils::SetScaled(float *source, PerseusImage* dest, int minScale, int maxScale, bool useScaling)
{
	unsigned char colourR = 240, colourG = 240, colourB = 240;

	int idx;

	dest->Clear(0);

	float minSource = 1000, maxSource = 0;
	for (idx = 1; idx < dest->width * dest->height; idx++)
	{
		if (source[idx] > 0)
		{
			if (source[idx] < minSource) minSource = source[idx];
			if (source[idx] > maxSource) maxSource = source[idx];
		}
	}

	float scale = ABS(float(maxScale) - float(minScale)) / ABS(float(maxSource) - float(minSource));

	for (idx = 0; idx < dest->width * dest->height; idx++)
	{ 
		if (source[idx] > 0)
		{
			unsigned char pixelValue = (unsigned char) ABS(source[idx] * scale);

			if (useScaling)
			{
				dest->pixels[idx * dest->bpp + 0] = unsigned char((colourR * pixelValue) / maxScale);
				dest->pixels[idx * dest->bpp + 1] = unsigned char((colourG * pixelValue) / maxScale);
				dest->pixels[idx * dest->bpp + 2] = unsigned char((colourB * pixelValue) / maxScale);
			}
			else
			{
				dest->pixels[idx * dest->bpp + 0] = maxScale;
				dest->pixels[idx * dest->bpp + 1] = maxScale;
				dest->pixels[idx * dest->bpp + 2] = maxScale;
			}
		}
	}
}