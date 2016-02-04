#ifndef __PERSEUS_VIEW_3D__
#define __PERSEUS_VIEW_3D__

#include "..\Others\PerseusDefines.h"

#include "..\Renderer\Transforms\CoordinateTransform.h"
#include "..\Renderer\Objects\Renderer3DView.h"

#include "..\Primitives\Vector3D.h"
#include "..\Primitives\Vector4D.h"

#include "..\Objects\View3DParams.h"

#include "..\Utils\ImageUtils.h"

using namespace Perseus::Objects;

using namespace Renderer::Primitives;
using namespace Renderer::Transforms;
using namespace Renderer::Objects;

using namespace Perseus::Utils;

namespace Perseus
{
	namespace Objects
	{
		class View3D
		{
		public:
			int viewId;

			float zBufferOffset;

			Renderer3DView* renderView;

			PerseusImage *depthMapFill;
			PerseusImage *depthMapWireframe;

			PerseusImage *imagePosteriors;
			PerseusImage *imageRegistered;
			PerseusImage *imageProximity;
			PerseusImage *imageSihluetteDT;

			void UpdateGPUFromCPU() {
				imageRegistered->UpdateGPUFromCPU();
			}

			void UpdateCPUFromGPU() {
				imageProximity->UpdateCPUFromGPU();
			}

			View3D(int viewIdx, char* cameraCalibFileName, int width, int height, View3DParams* params) {
				this->viewId = viewIdx;
				this->zBufferOffset = params->zBufferOffset;

				renderView = new Renderer3DView(width, height, cameraCalibFileName, params->zNear, params->zFar, viewIdx);

				//setup workspace images
				depthMapFill = new PerseusImage(width, height, PerseusImage::IMAGE_GRAY, true, true, true);
				depthMapWireframe = new PerseusImage(width, height, PerseusImage::IMAGE_GRAY, true, false, true);

				imageSihluetteDT = new PerseusImage(width, height, PerseusImage::IMAGE_GRAY, true, false, true);
				imagePosteriors = new PerseusImage(width, height, PerseusImage::IMAGE_GRAY, true, false, true);
				imageProximity = new PerseusImage(width, height, PerseusImage::IMAGE_RGBA, true, false, true);
				imageRegistered = new PerseusImage(width, height, PerseusImage::IMAGE_RGBA, true, false, true);
			}

			~View3D() {
				delete depthMapFill;
				delete depthMapWireframe;

				delete imageRegistered;
				delete imageSihluetteDT;
				delete imagePosteriors;
				delete imageProximity;

				delete renderView;
			}
		};
	}
}

#endif