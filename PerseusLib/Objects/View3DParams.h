#ifndef __PERSEUS_VIEW3D_PARAMS__
#define __PERSEUS_VIEW3D_PARAMS__

namespace Perseus
{
	namespace Objects
	{
		class View3DParams
		{
		public:
			float zNear, zFar;
			float zBufferOffset;

			View3DParams(void) {
				zBufferOffset = 0.0001f;
				zFar = 400.0f;
				zNear = 1.0f;
			}
			~View3DParams(void) {}
		};
	}
}

#endif