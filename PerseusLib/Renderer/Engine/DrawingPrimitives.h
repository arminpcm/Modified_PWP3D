#ifndef __RENDERER_DRAWING_PRIMITIVES__
#define __RENDERER_DRAWING_PRIMITIVES__

#include "..\..\Primitives\PerseusImage.h"
#include "..\..\Primitives\Vector2D.h"

#include "..\..\Others\PerseusDefines.h"

using namespace Perseus::Primitives;

namespace Renderer
{
	namespace Engine
	{
		class DrawingPrimitives
		{
			static DrawingPrimitives* instance;
		public:
			static DrawingPrimitives* Instance(void) {
				if (instance == NULL) instance = new DrawingPrimitives();
				return instance;
			}

			int sgn(int num) { if (num > 0) return(1); else if (num < 0) return(-1); else return(0); }

			void DrawLine(PerseusImage *image, int x1, int y1, int x2, int y2, VBYTE color);
			void DrawLineZ(PerseusImage *image, VFLOAT x0, VFLOAT y0, VFLOAT z0,
				VFLOAT x1, VFLOAT y1, VFLOAT z1, VINT meshId, VBYTE color);

			DrawingPrimitives(void);
			~DrawingPrimitives(void);
		};
	}
}

#endif