#ifndef __RENDERER_DRAWING_ENGINE__
#define __RENDERER_DRAWING_ENGINE__

#include "..\..\Primitives\PerseusImage.h"

#include "..\..\Objects\Pose3D.h"

#include "..\..\Renderer\Model\Model.h"
#include "..\..\Renderer\Model\ModelH.h"

#include "..\..\Renderer\Engine\DrawingPrimitives.h"

#include "..\..\Renderer\Transforms\CoordinateTransform.h"

#include "..\..\Renderer\Primitives\Quaternion.h"

#include "..\..\Renderer\Objects\Renderer3DObject.h"
#include "..\..\Renderer\Objects\Renderer3DView.h"

#include "..\..\CUDA\CUDAEngine.h"

using namespace Perseus::Primitives;
using namespace Perseus::Objects;

using namespace Renderer::Model3D;
using namespace Renderer::Primitives;
using namespace Renderer::Objects;
using namespace Renderer::Transforms;

#include <string>

namespace Renderer
{
	namespace Engine
	{
		class DrawingEngine
		{
		private:
			VECTOR3DA f;
			VFLOAT projectionMatrix[16], modelViewMatrix[16], pmMatrix[16], buffer[4];

			void applyCoordinateTransform(Renderer3DView* view, Renderer3DObject* object, float *pmMatrix);

			void drawFaceEdges(PerseusImage *image, ModelFace* currentFace, ModelH* drawingModel, VBYTE color, int* roiGenerated);
			void drawFaceFilled(PerseusImage *image, ModelFace* currentFace, ModelH* drawingModel, int objectId, VBYTE color, int meshId);
			void drawFaceFilled(PerseusImage *image, PerseusImage *depthMap, ModelFace* currentFace, ModelH* drawingModel, int objectId, VBYTE color, int meshId);
			void drawFaceZBufferUnnormalised(PerseusImage *image, ModelFace* currentFace, ModelH* drawingModel, int objectId, VBYTE color, int meshId);

			void drawWireframe(PerseusImage* imageWireframe, ModelH* drawingModel, int* roiGenerated);
			void drawFilled(PerseusImage* imageFill, ModelH* drawingModel, int objectId);
			void drawFilled(PerseusImage *imageFill, PerseusImage *depthMapFill, ModelH* drawingModel, int objectId);
			void drawZBufferUnnormalised(PerseusImage* imageFill, ModelH* drawingModel, int objectId);

			static DrawingEngine* instance;
		public:
			enum RenderingType {RENDERING_FILL, RENDERING_WIREFRAME, RENDERING_ZBUFFER};

			static DrawingEngine* Instance(void) {
				if (instance == NULL) instance = new DrawingEngine();
				return instance;
			}

			void Draw(Object3D* object, View3D *view, float* knownModelView, bool useCUDA, bool getBackData);
			void Draw(Object3D* object, View3D* view, Pose3D *pose, PerseusImage *imageFill, RenderingType renderingType);

			void ChangeROIWithBand(Object3D* object, View3D *view, int bandSize, int width, int height);
			void ChangeROIWithBand(View3D* view, int bandSize, int width, int height);

			void SetPMMatrices(Object3D* object, View3D* view, float *pmMatrix, float *invPMMatrix);

			DrawingEngine(void);
			~DrawingEngine(void);
		};
	}
}

#endif