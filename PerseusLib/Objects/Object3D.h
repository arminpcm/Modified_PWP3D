#ifndef __OBJECT_3D__
#define __OBJECT_3D__

#include "..\Others\PerseusDefines.h"

#include "..\Renderer\Transforms\CoordinateTransform.h"
#include "..\Renderer\Objects\Renderer3DObject.h"
#include "..\Renderer\Objects\Renderer3DView.h"

#include "..\Objects\HistogramVarBin.h"

#include "..\Primitives\Vector3D.h"
#include "..\Primitives\Vector4D.h"

#include "..\Objects\View3D.h"
#include "..\Objects\Pose3D.h"
#include "..\Objects\Object3DParams.h"
#include "..\Objects\View3DParams.h"
#include "..\Objects\HistogramVarBin.h"

using namespace Perseus::Objects;

using namespace Renderer::Primitives;
using namespace Renderer::Objects;
using namespace Renderer::Transforms;

namespace Perseus
{
	namespace Objects
	{
		class Object3D
		{
		public:
			int objectId;
			int viewCount;

			Pose3D *pose, *dpose, *initialPose;

			VFLOAT **invPMMatrix, **pmMatrix;

			VINT** roiGenerated; VINT** roiNormalised;

			Renderer3DObject* renderObject;

			HistogramVarBin **histogramVarBin;

			PerseusImage **imageHistogramMask, **imageWireframe, **imageFill, **imageSihluette, **imageCamera;

			PerseusImage **dt; PerseusImage **dtPosX, **dtPosY;
			PerseusImage **dtDX, **dtDY;

			void UpdatePoseFromRenderer(View3D* view) {
				pose->Set(&renderObject->objectCoordinateTransform[view->viewId]->translation,
					renderObject->objectCoordinateTransform[view->viewId]->rotation);
			}

			void UpdateRendererFromPose(View3D* view) {
				//TODO: combine with calibration to get render
				pose->Get(&renderObject->objectCoordinateTransform[view->viewId]->translation,
					renderObject->objectCoordinateTransform[view->viewId]->rotation);
			}

			Object3D(int objectId, int viewCount, char *objectFileName, int width, int height, Object3DParams* objectParams, View3DParams* viewParams)
			{
				int i;

				this->objectId = objectId;
				this->viewCount = viewCount;

				renderObject = new Renderer3DObject(objectFileName, viewCount, objectId);

				pose = new Pose3D();
				dpose = new Pose3D();
				initialPose = new Pose3D();

				initialPose->Set(objectParams->initialPose);

				histogramVarBin = new HistogramVarBin*[viewCount];

				invPMMatrix = new VFLOAT*[viewCount];
				pmMatrix = new VFLOAT*[viewCount];

				roiGenerated = new VINT*[viewCount];
				roiNormalised = new VINT*[viewCount];

				imageHistogramMask = new PerseusImage*[viewCount];
				imageSihluette = new PerseusImage*[viewCount];
				imageWireframe = new PerseusImage*[viewCount];
				imageFill = new PerseusImage*[viewCount];
				imageCamera = new PerseusImage*[viewCount];

				dt = new PerseusImage*[viewCount];
				dtPosX = new PerseusImage*[viewCount];
				dtPosY = new PerseusImage*[viewCount];
				dtDX = new PerseusImage*[viewCount];
				dtDY = new PerseusImage*[viewCount];

				for (i=0; i<viewCount; i++)
				{
					histogramVarBin[i] = new HistogramVarBin();
					histogramVarBin[i]->Set(objectParams->noVarBinHistograms, objectParams->noVarBinHistogramBins);

					cudaMallocHost((void**)&invPMMatrix[i], sizeof(VFLOAT) * 16);
					cudaMallocHost((void**)&pmMatrix[i], sizeof(VFLOAT) * 16);

					roiGenerated[i] = new VINT[6];
					roiNormalised[i] = new VINT[6];

					imageHistogramMask[i] = new PerseusImage(width, height, PerseusImage::IMAGE_GRAY, true, true, false);
					imageWireframe[i] = new PerseusImage(width, height, PerseusImage::IMAGE_GRAY, true, false, true);
					imageSihluette[i] = new PerseusImage(width, height, PerseusImage::IMAGE_GRAY, true, false, true);
					imageFill[i] = new PerseusImage(width, height, PerseusImage::IMAGE_GRAY, true, true, true);
					imageCamera[i] = new PerseusImage(width, height, PerseusImage::IMAGE_RGBA, true, false, true);

					dt[i] = new PerseusImage(width, height, PerseusImage::IMAGE_FLOAT, true, false, true);
					dtPosX[i] = new PerseusImage(width, height, PerseusImage::IMAGE_INT, true, false, true);
					dtPosY[i] = new PerseusImage(width, height, PerseusImage::IMAGE_INT, true, false, true);
					dtDX[i] = new PerseusImage(width, height, PerseusImage::IMAGE_FLOAT, true, false, true);
					dtDY[i] = new PerseusImage(width, height, PerseusImage::IMAGE_FLOAT, true, false, true);
				}
			}

			~Object3D(void)
			{
				delete renderObject;

				delete pose;
				delete dpose;
				delete initialPose;

				for (int i=0; i<viewCount; i++)
				{
					histogramVarBin[i]->Free();
					delete histogramVarBin[i];

					cudaFreeHost(invPMMatrix[i]);
					cudaFreeHost(pmMatrix[i]);

					delete roiGenerated[i];
					delete roiNormalised[i];

					delete imageHistogramMask[i];
					delete imageFill[i];
					delete imageSihluette[i];
					delete imageWireframe[i];
					delete imageCamera[i];

					delete dt[i];
					delete dtPosX[i];
					delete dtPosY[i];
					delete dtDX[i];
					delete dtDY[i];
				}

				delete histogramVarBin;

				delete invPMMatrix;
				delete pmMatrix;

				delete roiGenerated;
				delete roiNormalised;

				delete imageHistogramMask;
				delete imageFill;
				delete imageWireframe;
				delete imageSihluette;
				delete imageCamera;
				
				delete dt;
				delete dtPosX;
				delete dtPosY;
				delete dtDX;
				delete dtDY;
			}
		};
	}
}

#endif