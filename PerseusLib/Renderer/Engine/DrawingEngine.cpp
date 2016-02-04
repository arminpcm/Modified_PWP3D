#include "DrawingEngine.h"
#include <algorithm>
#include <math.h>
#include <omp.h>

#include "..\..\Utils\Timer.h"

#include "..\..\CUDA\CUDAEngine.h"

using namespace Perseus::Utils;

using namespace Renderer::Engine;
using namespace Renderer::Model3D;
using namespace Renderer::Primitives;
using namespace Renderer::Transforms;
using namespace Renderer::Objects;

using namespace std;

DrawingEngine* DrawingEngine::instance;

DrawingEngine::DrawingEngine(void) { }
DrawingEngine::~DrawingEngine(void) { }

inline int iround(float x) { 
	int t; 
	__asm 
	{
		fld x;
		fistp t;
	} 
	return t; 
}

template <class T>
inline T min3(T t1, T t2, T t3) 
{ T minim; minim = (t1 < t2) ? t1 : t2; minim = (t3 < minim) ? t3 : minim; return minim;}

template <class T>
inline T max3(T t1, T t2, T t3) 
{ T maxim; maxim = (t1 > t2) ? t1 : t2; maxim = (t3 > maxim) ? t3 : maxim; return maxim;}

void DrawingEngine::drawWireframe(PerseusImage* imageWireframe, ModelH* drawingModel, int* roiGenerated)
{
	size_t j;
	int i, localExtrems[4];

	ModelFace* currentFace;
	VBYTE currentColor;

	for (i=drawingModel->groups->size() - 1; i>=0; i--)
	{
		//currentColor = (i + 2) * (255 / (drawingModel->groups->size() + 1));
		currentColor = 254;

		for (j=0; j<(*drawingModel->groups)[i]->faces.size(); j++)
		{
			currentFace = (*drawingModel->groups)[i]->faces[j];

			if (currentFace->isVisible)
			{
				this->drawFaceEdges(imageWireframe, currentFace, drawingModel, currentColor, localExtrems);

				roiGenerated[0] = MIN(roiGenerated[0], localExtrems[0]);
				roiGenerated[1] = MIN(roiGenerated[1], localExtrems[1]);
				roiGenerated[2] = MAX(roiGenerated[2], localExtrems[2]);
				roiGenerated[3] = MAX(roiGenerated[3], localExtrems[3]);
			}
		}
	}

	roiGenerated[4] = roiGenerated[2] - roiGenerated[0] + 1;
	roiGenerated[5] = roiGenerated[3] - roiGenerated[1] + 1;

	this->drawFaceEdges(imageWireframe, currentFace, drawingModel, currentColor, localExtrems);
}

void DrawingEngine::drawFilled(PerseusImage* imageFill, ModelH* drawingModel, int objectId)
{
	size_t j;
	int i;

	VBYTE currentColor;

	for (i=drawingModel->groups->size() - 1; i>=0; i--)
	{
		currentColor = 128;

		for (j=0; j<(*drawingModel->groups)[i]->faces.size(); j++)
			this->drawFaceFilled(imageFill, (*drawingModel->groups)[i]->faces[j], drawingModel, objectId, currentColor, i);
	}
}

void DrawingEngine::drawFilled(PerseusImage *imageFill, PerseusImage *depthMapFill, ModelH* drawingModel, int objectId)
{
	size_t j;
	int i;

	VBYTE currentColor;

	for (i=drawingModel->groups->size() - 1; i>=0; i--)
	{
		//currentColor = (i + 1) * (255 / (drawingModel->groups->size() + 1));
		currentColor = 128;

		for (j=0; j<(*drawingModel->groups)[i]->faces.size(); j++)
			this->drawFaceFilled(imageFill, depthMapFill, (*drawingModel->groups)[i]->faces[j], drawingModel, objectId, currentColor, i);
	}
}

void DrawingEngine::drawZBufferUnnormalised(PerseusImage* imageFill, ModelH* drawingModel, int objectId)
{
	size_t j;
	int i;

	VBYTE currentColor;

	for (i=drawingModel->groups->size() - 1; i>=0; i--)
	{
		currentColor = 128;

		for (j=0; j<(*drawingModel->groups)[i]->faces.size(); j++)
			this->drawFaceZBufferUnnormalised(imageFill, (*drawingModel->groups)[i]->faces[j], drawingModel, objectId, currentColor, i);
	}
}

void DrawingEngine::Draw(Object3D* object, View3D* view, Pose3D *pose, PerseusImage *imageFill, RenderingType renderingType)
{
	int roi[6];

	Renderer3DObject *renderObject = object->renderObject;
	Renderer3DView *renderView = view->renderView;

	float modelViewMatrix[16], matrixFromSource[16], projectionMatrix[16], pmMatrix[16];

	modelViewMatrix[0] = 1; modelViewMatrix[5] = 1; modelViewMatrix[10] = 1; modelViewMatrix[15] = 1;
	modelViewMatrix[3] = 0; modelViewMatrix[7] = 0; modelViewMatrix[11] = 0;

	modelViewMatrix[12] = pose->translation->x; modelViewMatrix[13] = pose->translation->y; modelViewMatrix[14] = pose->translation->z;

	pose->rotation->GetMatrix(matrixFromSource);

	modelViewMatrix[0] = matrixFromSource[0]; modelViewMatrix[1] = matrixFromSource[1]; modelViewMatrix[2] = matrixFromSource[2];
	modelViewMatrix[4] = matrixFromSource[4]; modelViewMatrix[5] = matrixFromSource[5]; modelViewMatrix[6] = matrixFromSource[6];
	modelViewMatrix[8] = matrixFromSource[8]; modelViewMatrix[9] = matrixFromSource[9]; modelViewMatrix[10] = matrixFromSource[10];

	if (renderingType == DrawingEngine::RENDERING_ZBUFFER) renderView->cameraCoordinateTransform->GetProjectionMatrix(projectionMatrix, false);
	else renderView->cameraCoordinateTransform->GetProjectionMatrix(projectionMatrix, true);

	MathUtils::Instance()->SquareMatrixProduct(pmMatrix, projectionMatrix, modelViewMatrix, 4);

	this->applyCoordinateTransform(renderView, renderObject, pmMatrix);

	switch (renderingType)
	{
	case RENDERING_FILL:
		imageFill->Clear(0); //TODO remove me for speed
		imageFill->ClearZBuffer();
		drawFilled(imageFill, renderObject->drawingModel[view->viewId], object->objectId);
		break;
	case RENDERING_ZBUFFER:
		imageFill->ClearF(1000.0f);
		drawZBufferUnnormalised(imageFill, renderObject->drawingModel[view->viewId], object->objectId);
		break;
	case RENDERING_WIREFRAME:
		imageFill->Clear(0); //TODO remove me for speed
		drawWireframe(imageFill, renderObject->drawingModel[view->viewId], roi);
		break;
	}
}

void DrawingEngine::SetPMMatrices(Object3D *object, View3D *view, float *pmMatrix, float *invPMMatrix)
{
	memcpy(object->pmMatrix[view->viewId], pmMatrix, 16 * sizeof(float));
	memcpy(object->invPMMatrix[view->viewId], invPMMatrix, 16 * sizeof(float));
}

void DrawingEngine::Draw(Object3D* object, View3D* view, float* knownModelView, bool useCUDA, bool getBackData)
{
	Renderer3DObject *renderObject = object->renderObject;
	Renderer3DView *renderView = view->renderView;
	int objectId = object->objectId, viewId = view->viewId;

	float modelViewMatrix[16], projectionMatrix[16], pmMatrix[16], invPMMatrix[16];

	if (knownModelView == NULL)	renderObject->GetModelViewMatrix(modelViewMatrix, view->viewId);
	else { memcpy(modelViewMatrix, knownModelView, 16 * sizeof(float)); }

	renderView->cameraCoordinateTransform->GetProjectionMatrix(projectionMatrix, true);

	MathUtils::Instance()->SquareMatrixProduct(pmMatrix, projectionMatrix, modelViewMatrix, 4);
	MathUtils::Instance()->InvertMatrix4(invPMMatrix, pmMatrix);
	this->SetPMMatrices(object, view, pmMatrix, invPMMatrix);

	if (useCUDA)
	{
		renderObjectCUDA_SO(object, view);
		if (getBackData) getProcessedDataRendering(object, view);
	}
	else
	{
		int threadId;

		this->applyCoordinateTransform(renderView, renderObject, pmMatrix);

#pragma omp parallel num_threads(2) private(threadId)
		{
			threadId = omp_get_thread_num();

			switch (threadId)
			{
			case 0:
				object->imageWireframe[viewId]->Clear(0);
				object->roiGenerated[viewId][0] = 0xFFFF; object->roiGenerated[viewId][1] = 0xFFFF;
				object->roiGenerated[viewId][2] = -1; object->roiGenerated[viewId][3] = -1;

				drawWireframe(object->imageWireframe[viewId], renderObject->drawingModel[view->viewId], object->roiGenerated[viewId]);
				break;

			case 1:
				object->imageFill[viewId]->Clear(0); //TODO remove me for speed
				object->imageFill[viewId]->ClearZBuffer();

				drawFilled(object->imageFill[viewId], renderObject->drawingModel[view->viewId], objectId);
				break;
			}
		}
	}
}

void DrawingEngine::ChangeROIWithBand(Object3D* object, View3D *view, int bandSize, int width, int height)
{
	int *roiGenerated = object->roiGenerated[view->viewId];

	int roiTest[6];
	memcpy(roiTest, roiGenerated, 6 * sizeof(int));

	roiGenerated[0] = CLAMP(roiGenerated[0] - bandSize, 0, width);
	roiGenerated[1] = CLAMP(roiGenerated[1] - bandSize, 0, height);
	roiGenerated[2] = CLAMP(roiGenerated[2] + bandSize, 0, width);
	roiGenerated[3] = CLAMP(roiGenerated[3] + bandSize, 0, height);

	roiGenerated[4] = roiGenerated[2] - roiGenerated[0];
	roiGenerated[5] = roiGenerated[3] - roiGenerated[1];
}

void DrawingEngine::ChangeROIWithBand(View3D *view3D, int bandSize, int width, int height)
{
	Renderer3DView* view = view3D->renderView;

	int roiTest[6];
	memcpy(roiTest, view->roiGenerated, 6 * sizeof(int));

	view->roiGenerated[0] = CLAMP(view->roiGenerated[0] - bandSize, 0, width);
	view->roiGenerated[1] = CLAMP(view->roiGenerated[1] - bandSize, 0, height);
	view->roiGenerated[2] = CLAMP(view->roiGenerated[2] + bandSize, 0, width);
	view->roiGenerated[3] = CLAMP(view->roiGenerated[3] + bandSize, 0, height);

	view->roiGenerated[4] = view->roiGenerated[2] - view->roiGenerated[0];
	view->roiGenerated[5] = view->roiGenerated[3] - view->roiGenerated[1];
}

void DrawingEngine::drawFaceEdges(PerseusImage *image, ModelFace* currentFace, ModelH* drawingModel, VBYTE color, int* extrems)
{
	if (currentFace->verticesVectorCount != 3) return;

	VFLOAT x1 = drawingModel->verticesVectorPreP[currentFace->verticesVector[0]*4 + 0];
	VFLOAT y1 = drawingModel->verticesVectorPreP[currentFace->verticesVector[0]*4 + 1];
	VFLOAT z1 = drawingModel->verticesVectorPreP[currentFace->verticesVector[0]*4 + 2];

	VFLOAT x2 = drawingModel->verticesVectorPreP[currentFace->verticesVector[1]*4 + 0];
	VFLOAT y2 = drawingModel->verticesVectorPreP[currentFace->verticesVector[1]*4 + 1];
	VFLOAT z2 = drawingModel->verticesVectorPreP[currentFace->verticesVector[1]*4 + 2];

	VFLOAT x3 = drawingModel->verticesVectorPreP[currentFace->verticesVector[2]*4 + 0];
	VFLOAT y3 = drawingModel->verticesVectorPreP[currentFace->verticesVector[2]*4 + 1];
	VFLOAT z3 = drawingModel->verticesVectorPreP[currentFace->verticesVector[2]*4 + 2];

	//float c= x2 * (-y1+y3) + x3 * (-y2+y1) + x1 * ( y2 - y3 );
	//if (c < 0.003 || c > 0.004) return;
	//if ( c > 0) return;

	x1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 0];
	y1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 1];
	z1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 2];

	x2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 0];
	y2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 1];
	z2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 2];

	x3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 0];
	y3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 1];
	z3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 2];

	//if (z1 < 0)
	//{
	//DEBUGBREAK;
	//}

	//VFLOAT n1, n2, n3;
	//   n1 = (y1*z2)-(y2*z1);
	//   n2 = -(x1*z2)+(x2*z1);
	//   n3 = (x1*y2)-(y1*x2);

	//printf("%4.5f\n", n3);

	//float c = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3) ;
	//if (c < 0) return;
	//else printf("%f\n", c);

	x1 = CLAMP(x1, 0, (VFLOAT) image->width);
	y1 = CLAMP(y1, 0, (VFLOAT) image->height);
	x2 = CLAMP(x2, 0, (VFLOAT) image->width);
	y2 = CLAMP(y2, 0, (VFLOAT) image->height);
	x3 = CLAMP(x3, 0, (VFLOAT) image->width);
	y3 = CLAMP(y3, 0, (VFLOAT) image->height);

	DRAWLINE(image, x1, y1, x2, y2, color);
	DRAWLINE(image, x2, y2, x3, y3, color);
	DRAWLINE(image, x1, y1, x3, y3, color);

	extrems[0] = (VINT) x1;
	extrems[1] = (VINT) y1;
	extrems[2] = (VINT) x1;
	extrems[3] = (VINT) y1;

	extrems[0] = (VINT) MIN(extrems[0], x2);
	extrems[1] = (VINT) MIN(extrems[1], y2);
	extrems[2] = (VINT) MAX(extrems[2], x2);
	extrems[3] = (VINT) MAX(extrems[3], y2);

	extrems[0] = (VINT) MIN(extrems[0], x3);
	extrems[1] = (VINT) MIN(extrems[1], y3);
	extrems[2] = (VINT) MAX(extrems[2], x3);
	extrems[3] = (VINT) MAX(extrems[3], y3);

}

void DrawingEngine::drawFaceZBufferUnnormalised(PerseusImage *image, ModelFace* currentFace, ModelH* drawingModel, int objectId, VBYTE color, VINT meshId)
{
	if (currentFace->verticesVectorCount != 3) return;

	size_t i;
	size_t index;
	VFLOAT dx1, dx2, dx3, dz1, dz2, dz3, dxa, dxb, dza, dzb;
	VFLOAT dzX, Sz;

	VECTOR3DA S, E;
	VECTOR3DA A, B, C;
	VECTOR3DA orderedPoints[3];

	VFLOAT x1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 0];
	VFLOAT y1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 1];
	VFLOAT z1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 2];

	VFLOAT x2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 0];
	VFLOAT y2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 1];
	VFLOAT z2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 2];

	VFLOAT x3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 0];
	VFLOAT y3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 1];
	VFLOAT z3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 2];

	x1 = CLAMP(x1, 0, (VFLOAT) image->width);
	y1 = CLAMP(y1, 0, (VFLOAT) image->height);
	x2 = CLAMP(x2, 0, (VFLOAT) image->width);
	y2 = CLAMP(y2, 0, (VFLOAT) image->height);
	x3 = CLAMP(x3, 0, (VFLOAT) image->width);
	y3 = CLAMP(y3, 0, (VFLOAT) image->height);

	A = VECTOR3DA(x1, y1, z1);
	B = VECTOR3DA(x2, y2, z2);
	C = VECTOR3DA(x3, y3, z3);

	if (y1 < y2)
	{
		if (y3 < y1) { orderedPoints[0] = C; orderedPoints[1] = A; orderedPoints[2] = B; }
		else if (y3 < y2) { orderedPoints[0] = A; orderedPoints[1] = C; orderedPoints[2] = B; }
		else { orderedPoints[0] = A; orderedPoints[1] = B; orderedPoints[2] = C; }
	}
	else
	{
		if (y3 < y2) { orderedPoints[0] = C; orderedPoints[1] = B; orderedPoints[2] = A; }
		else if (y3 < y1) { orderedPoints[0] = B; orderedPoints[1] = C; orderedPoints[2] = A; }
		else { orderedPoints[0] = B; orderedPoints[1] = A;	orderedPoints[2] = C; }
	}

	A = orderedPoints[0]; B = orderedPoints[1]; C = orderedPoints[2];

	dx1 = (B.y - A.y) > 0 ? (B.x - A.x) / (B.y - A.y) : B.x - A.x;
	dx2 = (C.y - A.y) > 0 ? (C.x - A.x) / (C.y - A.y) : 0;
	dx3 = (C.y - B.y) > 0 ? (C.x - B.x) / (C.y - B.y) : 0;

	dz1 = (B.y - A.y) != 0 ? (B.z - A.z) / (B.y - A.y) : 0;
	dz2 = (C.y - A.y) != 0 ? (C.z - A.z) / (C.y - A.y) : 0;
	dz3 = (C.y - B.y) != 0 ? (C.z - B.z) / (C.y - B.y) : 0;

	S = E = A;

	B.y = floor(B.y - 0.5f); C.y = floor(C.y - 0.5f);

	if (dx1 > dx2) { dxa = dx2; dxb = dx1; dza = dz2; dzb = dz1; }
	else { dxa = dx1; dxb = dx2; dza = dz1; dzb = dz2; }

	for(; S.y <= B.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
	{
		dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
		Sz = S.z;

		for (i=(size_t)S.x; i<E.x; i++)
		{
			index = PIXELMATINDEX(i, S.y, image->width);
			if (Sz < image->pixelsF[index]) image->pixelsF[index] = Sz;
			Sz += dzX;
		}
	}

	if (dx1 > dx2) { dxa = dx2; dxb = dx3; dza = dz2; dzb = dz3; E = B; }
	else { dxa = dx3; dxb = dx2; dza = dz3; dzb = dz2; S = B; }

	for(; S.y <= C.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
	{
		dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
		Sz = S.z;

		for (i=(size_t)S.x; i<E.x; i++)
		{
			index = PIXELMATINDEX(i, S.y, image->width);
			if (Sz < image->pixelsF[index]) image->pixelsF[index] = Sz;
			Sz += dzX;
		}
	}
}


void DrawingEngine::drawFaceFilled(PerseusImage *image, ModelFace* currentFace, ModelH* drawingModel, int objectId, VBYTE color, VINT meshId)
{
	if (currentFace->verticesVectorCount != 3) return;

	size_t i;
	size_t index;
	VUINT intZ;
	VFLOAT dx1, dx2, dx3, dz1, dz2, dz3, dxa, dxb, dza, dzb;
	VFLOAT dzX, Sz;

	VECTOR3DA S, E;
	VECTOR3DA A, B, C;
	VECTOR3DA orderedPoints[3];

	VFLOAT x1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 0];
	VFLOAT y1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 1];
	VFLOAT z1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 2];

	VFLOAT x2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 0];
	VFLOAT y2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 1];
	VFLOAT z2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 2];

	VFLOAT x3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 0];
	VFLOAT y3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 1];
	VFLOAT z3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 2];

	x1 = CLAMP(x1, 0, (VFLOAT) image->width);
	y1 = CLAMP(y1, 0, (VFLOAT) image->height);
	x2 = CLAMP(x2, 0, (VFLOAT) image->width);
	y2 = CLAMP(y2, 0, (VFLOAT) image->height);
	x3 = CLAMP(x3, 0, (VFLOAT) image->width);
	y3 = CLAMP(y3, 0, (VFLOAT) image->height);

	A = VECTOR3DA(x1, y1, z1);
	B = VECTOR3DA(x2, y2, z2);
	C = VECTOR3DA(x3, y3, z3);

	if (y1 < y2)
	{
		if (y3 < y1) { orderedPoints[0] = C; orderedPoints[1] = A; orderedPoints[2] = B; }
		else if (y3 < y2) { orderedPoints[0] = A; orderedPoints[1] = C; orderedPoints[2] = B; }
		else { orderedPoints[0] = A; orderedPoints[1] = B; orderedPoints[2] = C; }
	}
	else
	{
		if (y3 < y2) { orderedPoints[0] = C; orderedPoints[1] = B; orderedPoints[2] = A; }
		else if (y3 < y1) { orderedPoints[0] = B; orderedPoints[1] = C; orderedPoints[2] = A; }
		else { orderedPoints[0] = B; orderedPoints[1] = A;	orderedPoints[2] = C; }
	}

	A = orderedPoints[0]; B = orderedPoints[1]; C = orderedPoints[2];

	dx1 = (B.y - A.y) > 0 ? (B.x - A.x) / (B.y - A.y) : B.x - A.x;
	dx2 = (C.y - A.y) > 0 ? (C.x - A.x) / (C.y - A.y) : 0;
	dx3 = (C.y - B.y) > 0 ? (C.x - B.x) / (C.y - B.y) : 0;

	dz1 = (B.y - A.y) != 0 ? (B.z - A.z) / (B.y - A.y) : 0;
	dz2 = (C.y - A.y) != 0 ? (C.z - A.z) / (C.y - A.y) : 0;
	dz3 = (C.y - B.y) != 0 ? (C.z - B.z) / (C.y - B.y) : 0;

	S = E = A;

	B.y = floor(B.y - 0.5f); C.y = floor(C.y - 0.5f);

	if (dx1 > dx2) { dxa = dx2; dxb = dx1; dza = dz2; dzb = dz1; }
	else { dxa = dx1; dxb = dx2; dza = dz1; dzb = dz2; }

	for(; S.y <= B.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
	{
		dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
		Sz = S.z;

		for (i=(size_t)S.x; i<E.x; i++)
		{
			index = PIXELMATINDEX(i, S.y, image->width);
			intZ = (unsigned int) (MAX_INT * Sz);

			if (intZ < image->zbuffer[index])
			{
				image->pixels[index] = 128;
				image->zbuffer[index] = intZ;
				image->objects[index] = objectId + 1;
			}

			if (intZ > image->zbufferInverse[index])
			{ image->zbufferInverse[index] = intZ; }

			Sz += dzX;
		}
	}

	if (dx1 > dx2) { dxa = dx2; dxb = dx3; dza = dz2; dzb = dz3; E = B; }
	else { dxa = dx3; dxb = dx2; dza = dz3; dzb = dz2; S = B; }

	for(; S.y <= C.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
	{
		dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
		Sz = S.z;

		for (i=(size_t)S.x; i<E.x; i++)
		{
			index = PIXELMATINDEX(i, S.y, image->width);
			intZ = (unsigned int) (MAX_INT * Sz);

			if (intZ < image->zbuffer[index])
			{
				image->pixels[index] = 128;
				image->zbuffer[index] = intZ;
				image->objects[index] = objectId + 1;
			}

			if (intZ > image->zbufferInverse[index])
			{ image->zbufferInverse[index] = intZ; }

			Sz += dzX;
		}
	}
}

void DrawingEngine::drawFaceFilled(PerseusImage *image, PerseusImage *depthMap, ModelFace* currentFace, ModelH* drawingModel, int objectId, VBYTE color, VINT meshId)
{
	size_t i;
	size_t index;
	VUINT intZ;
	VFLOAT dx1, dx2, dx3, dz1, dz2, dz3, dxa, dxb, dza, dzb;
	VFLOAT dzX, Sz;

	VECTOR3DA S, E;
	VECTOR3DA A, B, C;
	VECTOR3DA orderedPoints[3];

	VFLOAT x1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 0];
	VFLOAT y1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 1];
	VFLOAT z1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 2];

	VFLOAT x2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 0];
	VFLOAT y2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 1];
	VFLOAT z2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 2];

	VFLOAT x3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 0];
	VFLOAT y3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 1];
	VFLOAT z3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 2];

	x1 = CLAMP(x1, 0, (VFLOAT) image->width);
	y1 = CLAMP(y1, 0, (VFLOAT) image->height);
	x2 = CLAMP(x2, 0, (VFLOAT) image->width);
	y2 = CLAMP(y2, 0, (VFLOAT) image->height);
	x3 = CLAMP(x3, 0, (VFLOAT) image->width);
	y3 = CLAMP(y3, 0, (VFLOAT) image->height);

	A = VECTOR3DA(x1, y1, z1);
	B = VECTOR3DA(x2, y2, z2);
	C = VECTOR3DA(x3, y3, z3);

	if (y1 < y2)
	{
		if (y3 < y1) { orderedPoints[0] = C; orderedPoints[1] = A; orderedPoints[2] = B; }
		else if (y3 < y2) { orderedPoints[0] = A; orderedPoints[1] = C; orderedPoints[2] = B; }
		else { orderedPoints[0] = A; orderedPoints[1] = B; orderedPoints[2] = C; }
	}
	else
	{
		if (y3 < y2) { orderedPoints[0] = C; orderedPoints[1] = B; orderedPoints[2] = A; }
		else if (y3 < y1) { orderedPoints[0] = B; orderedPoints[1] = C; orderedPoints[2] = A; }
		else { orderedPoints[0] = B; orderedPoints[1] = A;	orderedPoints[2] = C; }
	}

	A = orderedPoints[0]; B = orderedPoints[1]; C = orderedPoints[2];

	dx1 = (B.y - A.y) > 0 ? (B.x - A.x) / (B.y - A.y) : B.x - A.x;
	dx2 = (C.y - A.y) > 0 ? (C.x - A.x) / (C.y - A.y) : 0;
	dx3 = (C.y - B.y) > 0 ? (C.x - B.x) / (C.y - B.y) : 0;

	dz1 = (B.y - A.y) != 0 ? (B.z - A.z) / (B.y - A.y) : 0;
	dz2 = (C.y - A.y) != 0 ? (C.z - A.z) / (C.y - A.y) : 0;
	dz3 = (C.y - B.y) != 0 ? (C.z - B.z) / (C.y - B.y) : 0;

	S = E = A;

	B.y = floor(B.y - 0.5f); C.y = floor(C.y - 0.5f);

	if (dx1 > dx2) { dxa = dx2; dxb = dx1; dza = dz2; dzb = dz1; }
	else { dxa = dx1; dxb = dx2; dza = dz1; dzb = dz2; }

	for(; S.y <= B.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
	{
		dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
		Sz = S.z;

		for (i=(size_t)S.x; i<E.x; i++)
		{
			index = PIXELMATINDEX(i, S.y, image->width);
			intZ = (unsigned int) (MAX_INT * Sz);

			if (intZ < image->zbuffer[index])
			{
				image->pixels[index] = 128;
				image->zbuffer[index] = intZ;
				image->objects[index] = objectId + 1;
			}

			if (intZ < depthMap->zbuffer[index])
			{
				depthMap->zbuffer[index] = intZ;
				depthMap->objects[index] = objectId + 1;
				depthMap->pixels[index] = (objectId + 1) * 32;
			}

			if (intZ > image->zbufferInverse[index])
			{ image->zbufferInverse[index] = intZ; }

			if (intZ > depthMap->zbufferInverse[index])
			{ depthMap->zbufferInverse[index] = intZ; }

			Sz += dzX;
		}
	}

	if (dx1 > dx2) { dxa = dx2; dxb = dx3; dza = dz2; dzb = dz3; E = B; }
	else { dxa = dx3; dxb = dx2; dza = dz3; dzb = dz2; S = B; }

	for(; S.y <= C.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
	{
		dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
		Sz = S.z;

		for (i=(size_t)S.x; i<E.x; i++)
		{
			index = PIXELMATINDEX(i, S.y, image->width);
			intZ = (unsigned int) (MAX_INT * Sz);

			if (intZ < image->zbuffer[index])
			{
				image->pixels[index] = 128;
				image->zbuffer[index] = intZ;
				image->objects[index] = objectId + 1;
			}

			if (intZ < depthMap->zbuffer[index])
			{
				depthMap->zbuffer[index] = intZ;
				depthMap->objects[index] = objectId + 1;
				depthMap->pixels[index] = (objectId + 1) * 32;
			}

			if (intZ > image->zbufferInverse[index])
			{ image->zbufferInverse[index] = intZ; }

			if (intZ > depthMap->zbufferInverse[index])
			{ depthMap->zbufferInverse[index] = intZ; }

			Sz += dzX;
		}
	}
}

void DrawingEngine::applyCoordinateTransform(Renderer3DView* view, Renderer3DObject* object, float *pmMatrix)
{
	size_t i;

	object->model->ToModelH(object->drawingModel[view->viewId]);

	for (i=0; i < object->drawingModel[view->viewId]->verticesVectorSize; i++)
	{
		VFLOAT* originalVertexAsDouble = &object->drawingModel[view->viewId]->originalVerticesVector[i*4];
		VFLOAT* vertexAsDouble = &object->drawingModel[view->viewId]->verticesVector[i*4];
		VFLOAT* vertexAsDoublePreP = &object->drawingModel[view->viewId]->verticesVectorPreP[i*4];

		MathUtils::Instance()->MatrixVectorProduct4(pmMatrix, originalVertexAsDouble, vertexAsDouble);
		MathUtils::Instance()->MatrixVectorProduct4(modelViewMatrix, originalVertexAsDouble, vertexAsDoublePreP);

		vertexAsDouble[0] = view->view[0] + view->view[2] * (vertexAsDouble[0] + 1)/2;
		vertexAsDouble[1] = view->view[1] + view->view[3] * (vertexAsDouble[1] + 1)/2;
		vertexAsDouble[2] = (vertexAsDouble[2] + 1)/2;

		if (view->camera3D->cameraType == Camera3D::PTAM_CAMERA)
		{
			MathUtils::Instance()->MatrixVectorProduct4(modelViewMatrix, originalVertexAsDouble, buffer);
			buffer[0] = buffer[0]/buffer[2];
			buffer[1] = buffer[1]/buffer[2];
			view->camera3D->Project(buffer, vertexAsDouble);
		}
	}
}


//void DrawingEngine::Draw(Renderer3DObject *object, Renderer3DView *view, float *knownModelView,
//						 PerseusImage *imageFill, PerseusImage *imageWireframe, int *roiGenerated, 
//						 PerseusImage *depthMapFill, PerseusImage *depthMapWireframe, int *roiGeneratedDepthMap)
//{
//	int threadId;
//
//	this->applyCoordinateTransform(view, object, knownModelView);
//
//#pragma omp parallel num_threads(2) private(threadId)
//	{
//		threadId = omp_get_thread_num();
//
//		switch (threadId)
//		{
//		case 0:
//			if (object->objectId == 0) 
//			{
//				depthMapWireframe->Clear(0);
//				roiGeneratedDepthMap[0] = 0xFFFF; roiGeneratedDepthMap[1] = 0xFFFF;
//				roiGeneratedDepthMap[2] = -1; roiGeneratedDepthMap[3] = -1;
//			}
//
//			imageWireframe->Clear(0);
//			roiGenerated[0] = 0xFFFF; roiGenerated[1] = 0xFFFF;
//			roiGenerated[2] = -1; roiGenerated[3] = -1;
//
//			drawWireframe(imageWireframe, object->drawingModel[view->viewId], roiGenerated);
//			drawWireframe(depthMapWireframe, object->drawingModel[view->viewId], roiGeneratedDepthMap);
//			break;
//
//		case 1:
//			if (object->objectId == 0)
//			{ depthMapFill->Clear(0); depthMapFill->ClearZBuffer(); }	
//
//			imageFill->Clear(0);
//			imageFill->ClearZBuffer();
//
//			//TODO depthmap
//			drawFilled(imageFill, depthMapFill, object->drawingModel[view->viewId], object->objectId);
//			break;
//		}
//	}
//}
