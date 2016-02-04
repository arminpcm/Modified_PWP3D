#ifndef __PERSEUS_IMAGE_UTILS__
#define __PERSEUS_IMAGE_UTILS__

#include "..\Others\PerseusDefines.h"
#include "..\Primitives\PerseusImage.h"

using namespace Perseus::Primitives;

namespace Perseus
{
	namespace Utils
	{
		class ImageUtils
		{
		private:
			static ImageUtils* instance;
		public:
			static ImageUtils* Instance(void) {
				if (instance == NULL) instance = new ImageUtils();
				return instance;
			}

			void SetScaled(float *source, PerseusImage* dest, int minScale, int maxScale, bool useScaling = true);
			void Overlay(PerseusImage* srcGrey, PerseusImage *destRGB, int destR = 255, int destG = 0, int destB = 0);
			void Copy(PerseusImage* src, PerseusImage *dest, bool includeZBuffer = false, bool tryOnlyPointer = false);
			void SaveImageToFile(PerseusImage* image, char* fileName);
			bool LoadImageFromFile(PerseusImage* image, char* fileName);

			ImageUtils(void);
			~ImageUtils(void);
		};
	}
}

#endif