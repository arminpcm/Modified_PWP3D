#ifndef __PERSEUS_I_ENERGY_FUNCTION__
#define __PERSEUS_I_ENERGY_FUNCTION__

#include "..\..\Objects\IterationConfiguration.h"
#include "..\..\Objects\Object3D.h"
#include "..\..\Objects\View3D.h"

#include "..\..\Renderer\Engine\DrawingEngine.h"

using namespace Perseus::Objects;
using namespace Renderer::Engine;

namespace Perseus
{
	namespace Optimiser
	{
		class IEnergyFunction
		{
		public:
			virtual void GetFirstDerivativeValues(Object3D **objects, int objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig) = 0;
			virtual void PrepareIteration(Object3D **objects, int objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig) = 0;

			IEnergyFunction(void) { }
			virtual ~IEnergyFunction(void) { }
		};
	}
}

#endif 