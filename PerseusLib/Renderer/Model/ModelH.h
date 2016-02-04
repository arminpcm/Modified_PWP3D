#ifndef __RENDERER_MODEL_H__
#define __RENDERER_MODEL_H__

#include "..\..\Others\PerseusDefines.h"

#include <stdio.h>

#include "..\Model\ModelGroup.h"
#include "..\Model\ModelVertexH.h"

namespace Renderer
{
	namespace Model3D
	{
		class ModelH
		{
		public:
			bool isAllocated;
			std::vector<ModelGroup*>* groups;
			
			VFLOAT *verticesVector;
			VFLOAT *verticesVectorPreP;
			VFLOAT *originalVerticesVector;
			size_t verticesVectorSize;

			float* verticesGPU, *verticesGPUBuff;
			
			VFLOAT minZ;

			int faceCount;

			ModelH(void);
			~ModelH(void);
		};
	}
}

#endif