#include "EFSingleObject.h"

using namespace Perseus::Optimiser;

#include "..\..\CUDA\CUDAEngine.h"

#include "..\..\Utils\ImageUtils.h"
using namespace Perseus::Utils;


EFSingleObject::EFSingleObject(void)
{
}

EFSingleObject::~EFSingleObject(void)
{
}

void EFSingleObject::PrepareIteration(Object3D **objects, int objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig)
{
	Object3D* object = objects[0];
	View3D* view = views[0];

	int objectId = object->objectId, viewId = view->viewId;

	DrawingEngine::Instance()->Draw(object, view, NULL, iterConfig->useCUDARender, !iterConfig->useCUDAEF);
	DrawingEngine::Instance()->ChangeROIWithBand(object, view, iterConfig->levelSetBandSize, iterConfig->width, iterConfig->height);

	registerObjectImage(object, view, iterConfig->useCUDARender);
	if (iterConfig->useCUDAEF) registerObjectAndViewGeometricData(object, view);

	processDTSihluetteLSDXDY(object, view, iterConfig->levelSetBandSize);
}

void EFSingleObject::GetFirstDerivativeValues(Object3D **objects, int objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig)
{
	Object3D* object = objects[0];
	View3D* view = views[0];

	if (iterConfig->useCUDAEF)
	{
		processAndGetEFFirstDerivatives(object, view);
		return;
	}
		
	getProcessedDataDTSihluetteLSDXDY(object, view);

	int objectId = object->objectId, viewId = view->viewId, width = iterConfig->width, height = iterConfig->height;

	int i, j, k, idx, idxB, icX, icY, icZ, hidx; 
	float pYB, pYF, dtIdx, dfPPGeneric, dirac, heaviside;
	unsigned char r, b, g;

	float *dt = object->dt[viewId]->pixelsF;
	int *dtPosX = object->dtPosX[viewId]->pixelsI, *dtPosY = object->dtPosY[viewId]->pixelsI;
	float *dtDX = object->dtDX[viewId]->pixelsF, *dtDY = object->dtDY[viewId]->pixelsF;

	float xProjected[4], xUnprojected[4], xUnrotated[4], dfPP[7], dpose[7], otherInfo[2];

	for (i=0; i<7; i++) dpose[i] = 0;

	for (j=0, idx=0, idxB=0; j<height; j++) for (i=0; i<width; idx++, idxB+=view->imageRegistered->bpp, i++)
	{
		dtIdx = dt[idx];
		if (dtPosY[idx] >= 0)
		{
			icX = i; icY = j;
			if (dtIdx < 0) { icX = dtPosX[idx] + object->roiGenerated[viewId][0]; icY = dtPosY[idx] + object->roiGenerated[viewId][1]; }
			icZ = icX + icY * width;

			hidx = int(4096 + 512 * dtIdx);
			if (hidx >= 0 && hidx < MathUtils::Instance()->heavisideSize)
			{
				heaviside = MathUtils::Instance()->heavisideFunction[hidx];

				r = view->imageRegistered->pixels[idxB + 0]; g = view->imageRegistered->pixels[idxB + 1]; b = view->imageRegistered->pixels[idxB + 2];

				object->histogramVarBin[viewId]->GetValue(&pYF, &pYB, r, g, b, i, j);

				pYF += 0.0000001f; pYB += 0.0000001f;

				dirac = (1.0f / float(PI)) * (1 / (dtIdx * dtIdx + 1.0f) + float(1e-3));
				dfPPGeneric = dirac * (pYF - pYB) / (heaviside * (pYF - pYB) + pYB);

				// run 1
				xProjected[0] = (float) 2 * (icX - view->renderView->view[0]) / view->renderView->view[2] - 1;
				xProjected[1] = (float) 2 * (icY - view->renderView->view[1]) / view->renderView->view[3] - 1;
				xProjected[2] = (float) 2 * ((float)object->imageFill[viewId]->zbuffer[icZ] / (float)MAX_INT) - 1;
				xProjected[3] = 1;

				MathUtils::Instance()->MatrixVectorProduct4(view->renderView->invP, xProjected, xUnprojected);
				MathUtils::Instance()->MatrixVectorProduct4(object->invPMMatrix[viewId], xProjected, xUnrotated);

				otherInfo[0] = view->renderView->projectionParams.A * dtDX[idx];
				otherInfo[1] = view->renderView->projectionParams.B * dtDY[idx];

				dfPP[0] = -otherInfo[0] / xUnprojected[2]; 
				dfPP[1] = -otherInfo[1] / xUnprojected[2];
				dfPP[2] = (otherInfo[0] * xUnprojected[0] + otherInfo[1] * xUnprojected[1]) / (xUnprojected[2] * xUnprojected[2]);

				object->renderObject->objectCoordinateTransform[viewId]->rotation->GetDerivatives(dfPP + 3, xUnprojected, xUnrotated, 
					view->renderView->projectionParams.all, otherInfo);

				for (k=0; k<7; k++) { dfPP[k] *= dfPPGeneric; dpose[k] += dfPP[k]; }

				// run 2
				xProjected[0] = (float) 2 * (icX - view->renderView->view[0]) / view->renderView->view[2] - 1;
				xProjected[1] = (float) 2 * (icY - view->renderView->view[1]) / view->renderView->view[3] - 1;
				xProjected[2] = (float) 2 * (float(object->imageFill[viewId]->zbufferInverse[icZ]) / float(MAX_INT)) - 1;
				xProjected[3] = 1;

				MathUtils::Instance()->MatrixVectorProduct4(view->renderView->invP, xProjected, xUnprojected);
				MathUtils::Instance()->MatrixVectorProduct4(object->invPMMatrix[viewId], xProjected, xUnrotated);

				otherInfo[0] = view->renderView->projectionParams.A * dtDX[idx];
				otherInfo[1] = view->renderView->projectionParams.B * dtDY[idx];

				dfPP[0] = -otherInfo[0] / xUnprojected[2];
				dfPP[1] = -otherInfo[1] / xUnprojected[2];
				dfPP[2] = (otherInfo[0] * xUnprojected[0] + otherInfo[1] * xUnprojected[1]) / (xUnprojected[2] * xUnprojected[2]);

				object->renderObject->objectCoordinateTransform[viewId]->rotation->GetDerivatives(dfPP + 3, xUnprojected, xUnrotated, 
					view->renderView->projectionParams.all, otherInfo);

				for (k=0; k<7; k++) { dfPP[k] *= dfPPGeneric; dpose[k] += dfPP[k]; }
			}
		}
	}

	object->dpose->Set(dpose);
}