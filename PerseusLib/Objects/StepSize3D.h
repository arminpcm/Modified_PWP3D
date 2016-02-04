#ifndef __PERSEUS_STEPSIZE3D__
#define __PERSEUS_STEPSIZE3D__

namespace Perseus
{
	namespace Objects
	{
		class StepSize3D
		{
		public:
			float tXY, tZ, r;

			StepSize3D(void) { tXY = 0; tZ = 0; r = 0; }
			StepSize3D(float r, float tXY, float tZ) { this->tXY = tXY; this->tZ = tZ; this->r = r; }

			~StepSize3D(void) { }
		};
	}
}

#endif