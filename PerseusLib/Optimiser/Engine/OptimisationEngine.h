#ifndef __PERSEUS_OPTIMISATION_ENGINE__
#define __PERSEUS_OPTIMISATION_ENGINE__

#include "..\..\Primitives\PerseusImage.h"
#include "..\..\Objects\IterationConfiguration.h"
#include "..\..\Objects\Object3D.h"
#include "..\..\Objects\View3D.h"
#include "..\..\Objects\StepSize3D.h"
#include "..\..\Optimiser\EFs\IEnergyFunction.h"
#include "..\..\Optimiser\EFs\EFSingleObject.h"

using namespace Perseus::Primitives;
using namespace Perseus::Objects;

namespace Perseus
{
	namespace Optimiser
	{
		class OptimisationEngine
		{
		private:
			static OptimisationEngine* instance;

			IEnergyFunction *energyFunction;
			IEnergyFunction *energyFunctionSO;

			IterationConfiguration *iterConfig;

			int objectCount, viewCount;
			Object3D** objects;
			View3D** views;
			
			StepSize3D **stepSizes;

			bool HasConverged();

			void SetStepSizes(StepSize3D *stepSizePreset);

			void NormaliseRotation();

			void DescendWithGradient(StepSize3D *stepSize, IterationTarget iterTarget);
			void AdvanceTranslation(Object3D* object, float stepSizeXY, float stepSizeZ);
			void AdvanceRotation(Object3D* object, float stepSizeR);

			void RunOneMultiIteration(IterationTarget iterTarget);
			void RunOneSingleIteration(StepSize3D* stepSize, IterationTarget iterTarget);

		public:
			static OptimisationEngine* Instance(void) {
				if (instance == NULL) instance = new OptimisationEngine();
				return instance;
			}

			void Initialise(int width, int heigh);
			void Shutdown();

			void RegisterViewImage(View3D *view, PerseusImage* image);
			void Minimise(Object3D **objects, View3D **views, IterationConfiguration *iterConfig);
			void GetImage(PerseusImage* image, GetImageType getImageType, Object3D* object, View3D* view);

			OptimisationEngine(void);
			~OptimisationEngine(void);
		};
	}
}

#endif
