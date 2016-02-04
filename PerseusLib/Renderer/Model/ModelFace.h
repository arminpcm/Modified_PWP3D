#ifndef __RENDERER_MODEL_FACE__
#define __RENDERER_MODEL_FACE__

#include <vector>

#include "..\..\Others\PerseusDefines.h"
#include "..\..\Renderer\Model\ModelVertex.h"

namespace Renderer
{
	namespace Model3D
	{
		class ModelFace
		{
		public:
			std::vector<int> vertices;
			int *verticesVector;
			size_t verticesVectorCount;
			size_t edgeListStartIndex;
			size_t edgeListStopIndex;
			int edgeCount;

			VBOOL isInvisible;
			VBOOL isVisible;

			ModelFace(void) {}
			~ModelFace(void) { delete verticesVector; }
		};
	}
}

#endif