#ifndef __PERSEUS_OBJECT3D_PARAMS__
#define __PERSEUS_OBJECT3D_PARAMS__

#include "..\Objects\Pose3D.h"

namespace Perseus
{
	namespace Objects
	{
		class Object3DParams
		{
		public:
			Pose3D* initialPose;

			int numberOfOptimizedVariables;
			int noVarBinHistograms; // max 8
			int noVarBinHistogramBins[4];

			Object3DParams(float tx, float ty, float tz, float rx, float ry, float rz) {
				noVarBinHistogramBins[0] = 8;
				noVarBinHistogramBins[1] = 16;
				noVarBinHistogramBins[2] = 32;
				noVarBinHistogramBins[3] = 64;
				noVarBinHistograms = 4;
				numberOfOptimizedVariables = 7;

				initialPose = new Pose3D(tx, ty, tz, rx, ry, rz);
			}
			~Object3DParams(void) 
			{ 
				delete initialPose; 
			}
		};
	}
}

#endif