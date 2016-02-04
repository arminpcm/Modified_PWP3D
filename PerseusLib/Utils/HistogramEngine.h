#ifndef __VISION_ENGINE__
#define __VISION_ENGINE__

#include "..\Primitives\PerseusImage.h"
#include "..\Utils\ImageUtils.h"

#include "..\Objects\HistogramVarBin.h"
#include "..\Objects\Object3D.h"
#include "..\Objects\View3D.h"
#include "..\Objects\Pose3D.h"

#include "..\Renderer\Engine\DrawingEngine.h"

using namespace Perseus::Primitives;
using namespace Perseus::Objects;
using namespace Perseus::Utils;

using namespace Renderer::Engine;

namespace Perseus
{
	namespace Utils
	{
		class HistogramEngine
		{
		private:
			static HistogramEngine* instance;

			void NormaliseHistogramVarBin(Perseus::Objects::HistogramVarBin *histogram);
			void BuildHistogramVarBin(Perseus::Objects::HistogramVarBin *histogram, PerseusImage *mask, PerseusImage* image, int objectId);
			void BuildHistogramVarBin(Perseus::Objects::HistogramVarBin *histogram, bool *mask, PerseusImage* image, int objectId);
			void BuildHistogramVarBin(Perseus::Objects::HistogramVarBin *histogram, PerseusImage *mask, float* dt, PerseusImage* image, int objectId);

			void UpdateVarBinHistogram(PerseusImage* originalImage, PerseusImage* mask, Perseus::Objects::HistogramVarBin *histogram, int objectId);
			void UpdateVarBinHistogram(PerseusImage* originalImage, PerseusImage* mask, float* dt, Perseus::Objects::HistogramVarBin *histogram, int objectId);
		public:
			static HistogramEngine* Instance(void) {
				if (instance == NULL) instance = new HistogramEngine();
				return instance;
			}

			void UpdateVarBinHistogram(Object3D* object, View3D* view, PerseusImage* originalImage, PerseusImage* mask);
			void UpdateVarBinHistogram(Object3D* object, View3D* view, PerseusImage* originalImage, bool* mask);
			void UpdateVarBinHistogram(Object3D* object, View3D* view, PerseusImage* originalImage, Pose3D* pose);

			void SetVarBinHistogram(Object3D* object, View3D* view, float2 *normalised);

			HistogramEngine(void);
			~HistogramEngine(void);
		};
	}
}

#endif