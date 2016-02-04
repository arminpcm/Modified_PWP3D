#ifndef __PERSEUS_ITERATION_CONFIGURATION__
#define __PERSEUS_ITERATION_CONFIGURATION__

#include "..\Objects\Pose3D.h"
#include "..\Objects\StepSize3D.h"
#include "..\Others\PerseusDefines.h"

using namespace Perseus::Objects;

#ifndef PERSEUS_MAX_OBJECT_COUNT
#define PERSEUS_MAX_OBJECT_COUNT 100
#endif

namespace Perseus
{
	namespace Objects
	{
		class IterationConfiguration
		{
		public:
			int iterCount;

			int width;
			int height;

			int levelSetBandSize;

			int objectCount;
			int viewCount;

			int iterObjectsId[100];
			int iterViewsId[100];

			IterationTarget iterTarget[100];
			StepSize3D *stepSize;

			bool useCUDARender;
			bool useCUDAEF;

			IterationConfiguration(void) { 
				int i;
				objectCount = 0; viewCount = 0; iterCount = 1; stepSize = 0;
				for (i=0; i<PERSEUS_MAX_OBJECT_COUNT; i++) iterTarget[i] = ITERATIONTARGET_BOTH;
				useCUDARender = false;
				useCUDAEF = false;
			}
			~IterationConfiguration(void) { 
				if (stepSize != 0) delete stepSize;
			} 
		};
	}
}

#endif