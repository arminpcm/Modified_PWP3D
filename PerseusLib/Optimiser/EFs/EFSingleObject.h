#ifndef __PERSEUS_EF_SINGLE_OBJECT__
#define __PERSEUS_EF_SINGLE_OBJECT__

#include "..\..\Optimiser\EFs\IEnergyFunction.h"

namespace Perseus
{
	namespace Optimiser
	{
		class EFSingleObject: public IEnergyFunction
		{
		public:
			void PrepareIteration(Object3D **objects, int objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig);
			void GetFirstDerivativeValues(Object3D **objects, int objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig);

			EFSingleObject(void);
			~EFSingleObject(void);
		};
	}
}

#endif