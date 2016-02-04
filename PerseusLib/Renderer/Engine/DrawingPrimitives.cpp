// Extremely Fast Line Algorithm Var E (Addition Fixed Point PreCalc)
// Copyright 2001-2, By Po-Han Lin


// Freely useable in non-commercial applications as long as credits 
// to Po-Han Lin and link to http://www.edepot.com is provided in 
// source code and can been seen in compiled executable.  
// Commercial applications please inquire about licensing the algorithms.
//
// Lastest version at http://www.edepot.com/phl.html
// This version is for standard displays (up to 65536x65536)
// For small display version (256x256) visit http://www.edepot.com/lineex.html

#include "DrawingPrimitives.h"
#include <math.h>

using namespace Renderer::Engine;

DrawingPrimitives* DrawingPrimitives::instance;

DrawingPrimitives::DrawingPrimitives(void)
{
}

DrawingPrimitives::~DrawingPrimitives(void)
{
}

void DrawingPrimitives::DrawLine(PerseusImage *image, int x, int y, int x2, int y2, VBYTE color)
{
	VBOOL yLonger=false;
	int shortLen=y2-y;
	int longLen=x2-x;
	if (abs(shortLen)>abs(longLen)) 
	{
		int swap=shortLen;
		shortLen=longLen;
		longLen=swap;				
		yLonger=true;
	}
	int decInc;
	if (longLen==0) decInc=0;
	else decInc = (shortLen << 16) / longLen;

	if (yLonger) 
	{
		if (longLen>0)
		{
			longLen+=y;
			for (int j=0x8000+(x<<16);y<=longLen;++y)
			{
				GETPIXEL(image,j >> 16,y) = color;	
				j+=decInc;
			}
			return;
		}
		longLen+=y;
		for (int j=0x8000+(x<<16);y>=longLen;--y) 
		{
			GETPIXEL(image,j >> 16,y) = color;	
			j-=decInc;
		}
		return;	
	}

	if (longLen>0) 
	{
		longLen+=x;
		for (int j=0x8000+(y<<16);x<=longLen;++x)
		{
			GETPIXEL(image,x,j >> 16) = color;
			j+=decInc;
		}
		return;
	}
	longLen+=x;
	for (int j=0x8000+(y<<16);x>=longLen;--x) 
	{
		GETPIXEL(image,x,j >> 16) = color;
		j-=decInc;
	}
}

void DrawingPrimitives::DrawLineZ(PerseusImage *image, VFLOAT x0, VFLOAT y0, VFLOAT z0, 
								  VFLOAT x1, VFLOAT y1, VFLOAT z1, VINT meshId, VBYTE color)
{
	VFLOAT oz0 = z0, oz1 = z1;
	VFLOAT oy0 = y0, ox0 = x0;
	VFLOAT z0x = z0, z0y = z0;
	VFLOAT stepzX, stepzY;
	VFLOAT z0d;

	int ix0 = (int) x0; int iy0 = (int) y0;
	int ix1 = (int) x1; int iy1 = (int) y1;
	int dy = (int)y1 - (int)y0;
	int dx = (int)x1 - (int)x0;
	int istepx, istepy;

	int index;
	int signumX, signumY, signumZ;
	signumX = (dx < 0) ? -1 : 1;
	signumY = (dy < 0) ? -1 : 1;
	signumZ = (z1 - z0) < 0 ? -1 : 1;

	if (dy < 0) { dy = -dy;  istepy = -1; } else { istepy = 1; }
	if (dx < 0) { dx = -dx;  istepx = -1; } else { istepx = 1; }
	dy <<= 1;
	dx <<= 1;

	stepzX = (x1 - x0) != 0 ? (z1 - z0) / (x1 - x0) : 0;
	stepzY = (y1 - y0) != 0 ? (z1 - z0) / (y1 - y0) : 0;

	index = PIXELMATINDEX(ix0, iy0, image->width);
	if (z0 < image->zbuffer[index])
	{
		image->pixels[index] = color;
		image->zbuffer[index] = (unsigned int) z0 * MAX_INT;
	}

	if (dx > dy)
	{
		int fraction = dy - (dx >> 1);
		while (ix0 != ix1)
		{
			if (fraction >= 0)
			{
				iy0 += istepy;
				z0y += (y0 - oy0) * stepzY;
				fraction -= dx;
			}

			ix0 += istepx;
			x0 += istepx;
			z0d = z0y + (x0 - ox0) * stepzX;
			fraction += dy;

			index = PIXELMATINDEX(ix0, iy0, image->width);
			if (z0d < image->zbuffer[index])
			{
				image->pixels[index] = color;
				image->zbuffer[index] = (unsigned int) z0d * MAX_INT;
			}
		}
	}
	else
	{
		int fraction = dx - (dy >> 1);
		while (iy0 != iy1)
		{
			if (fraction >= 0)
			{
				ix0 += istepx;
				z0x += (x0 - ox0) * stepzX;
				fraction -= dy;
			}

			iy0 += istepy;
			y0 += istepy;
			z0d = z0x + (y0 - oy0) * stepzY;
			fraction += dx;

			index = PIXELMATINDEX(ix0, iy0, image->width);
			if (z0d < image->zbuffer[index])
			{
				image->pixels[index] = color;
				image->zbuffer[index] = (unsigned int) z0d * MAX_INT;
			}
		}
	}
}