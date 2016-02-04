#ifndef __RENDERER_MODEL_GROUP__
#define __RENDERER_MODEL_GROUP__

#include "..\..\Others\PerseusDefines.h"

#include "..\..\Renderer\Model\ModelFace.h"
#include "..\..\Renderer\Model\ModelVertex.h"

//#include <vector>

namespace Renderer
{
	namespace Model3D
	{
		class ModelGroup
		{
		public:

			std::vector<ModelFace*> faces;
			char* groupName;

			ModelGroup(char* groupName);
			ModelGroup(void);
			~ModelGroup(void);
		};
	}
}

#endif