#include "OptimisationEngine.h"

#include "..\..\CUDA\CUDAEngine.h"

using namespace Perseus::Optimiser;

OptimisationEngine* OptimisationEngine::instance;

OptimisationEngine::OptimisationEngine(void) { }
OptimisationEngine::~OptimisationEngine(void) { }

void OptimisationEngine::Initialise(int width, int height)
{
	int i;

	objects = new Object3D*[100];
	views = new View3D*[100];

	stepSizes = new StepSize3D*[8];
	for (i=0; i<8; i++) stepSizes[i] = new StepSize3D();

	energyFunctionSO = new EFSingleObject();

	MathUtils::Instance()->ReadAndAllocateHeaviside(8192, "Files/Others/heaviside.txt");

	initialiseCUDA(width, height, MathUtils::Instance()->heavisideFunction, MathUtils::Instance()->heavisideSize);
}

void OptimisationEngine::Shutdown()
{
	int i;

	shutdownCUDA();

	for (i=0; i<8; i++) delete stepSizes[i];

	delete stepSizes;
	delete objects;
	delete views;
	delete energyFunctionSO;

	MathUtils::Instance()->DeallocateHeaviside();

	delete instance;
}

void OptimisationEngine::SetStepSizes(StepSize3D *stepSizePreset)
{
	stepSizes[0]->tXY = -0.005f * stepSizePreset->tXY; stepSizes[0]->tZ = -0.005f * stepSizePreset->tZ; stepSizes[0]->r = -0.0008f * stepSizePreset->r;

	stepSizes[1]->tXY = -0.003f * stepSizePreset->tXY; stepSizes[1]->tZ = -0.003f * stepSizePreset->tZ; stepSizes[1]->r = -0.0003f * stepSizePreset->r;
	stepSizes[2]->tXY = -0.003f * stepSizePreset->tXY; stepSizes[2]->tZ = -0.003f * stepSizePreset->tZ; stepSizes[2]->r = -0.0003f * stepSizePreset->r;
	stepSizes[3]->tXY = -0.003f * stepSizePreset->tXY; stepSizes[3]->tZ = -0.003f * stepSizePreset->tZ; stepSizes[3]->r = -0.0003f * stepSizePreset->r;

	stepSizes[4]->tXY = -0.002f * stepSizePreset->tXY; stepSizes[4]->tZ = -0.003f * stepSizePreset->tZ; stepSizes[4]->r = -0.0003f * stepSizePreset->r;
	stepSizes[5]->tXY = -0.002f * stepSizePreset->tXY; stepSizes[5]->tZ = -0.003f * stepSizePreset->tZ; stepSizes[5]->r = -0.0003f * stepSizePreset->r;
	stepSizes[6]->tXY = -0.002f * stepSizePreset->tXY; stepSizes[6]->tZ = -0.003f * stepSizePreset->tZ; stepSizes[6]->r = -0.0003f * stepSizePreset->r;
	stepSizes[7]->tXY = -0.002f * stepSizePreset->tXY; stepSizes[7]->tZ = -0.003f * stepSizePreset->tZ; stepSizes[7]->r = -0.0003f * stepSizePreset->r;
}

void OptimisationEngine::RegisterViewImage(View3D *view, PerseusImage* image)
{
	ImageUtils::Instance()->Copy(image, view->imageRegistered);
	view->imageRegistered->UpdateGPUFromCPU();
}

void OptimisationEngine::Minimise(Object3D **objects, View3D **views, IterationConfiguration *iterConfig)
{
	int i, j;

	this->iterConfig = iterConfig;

	objectCount = iterConfig->objectCount;
	viewCount = iterConfig->viewCount;

	this->SetStepSizes(iterConfig->stepSize);

	for (i=0; i<objectCount; i++) this->objects[i] = objects[iterConfig->iterObjectsId[i]];
	for (i=0; i<viewCount; i++) this->views[i] = views[iterConfig->iterViewsId[i]];

	for (i=0; i<objectCount; i++)
	{
		this->objects[i]->initialPose->CopyInto(this->objects[i]->pose);
		for (j=0; j<viewCount; j++) this->objects[i]->UpdateRendererFromPose(views[j]);
	}

	if (objectCount == 1 && viewCount == 1) energyFunction = energyFunctionSO;
	else exit(1);

	for (i=0; i<iterConfig->iterCount; i++) this->RunOneMultiIteration(iterConfig->iterTarget[i]);
}

void OptimisationEngine::RunOneMultiIteration(IterationTarget iterTarget)
{
	this->RunOneSingleIteration(stepSizes[0], iterTarget); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[1], iterTarget); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[2], iterTarget); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[3], iterTarget); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[4], iterTarget); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[5], iterTarget); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[6], iterTarget); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[7], iterTarget); if (this->HasConverged()) return;

	this->NormaliseRotation();
}

void OptimisationEngine::RunOneSingleIteration(StepSize3D* stepSize, IterationTarget iterTarget)
{
	energyFunction->PrepareIteration(objects, objectCount, views, viewCount, iterConfig);
	energyFunction->GetFirstDerivativeValues(objects, objectCount, views, viewCount, iterConfig);

	this->DescendWithGradient(stepSize, iterTarget);
}

void OptimisationEngine::DescendWithGradient(StepSize3D *stepSize, IterationTarget iterTarget)
{
	int i, j;

	for (i = 0; i < objectCount; i++) for (j = 0; j < viewCount; j++)
	{
		switch (iterTarget)
		{
		case ITERATIONTARGET_BOTH:
			AdvanceTranslation(objects[i], stepSize->tXY, stepSize->tZ); 
			AdvanceRotation(objects[i], stepSize->r);
			break;
		case ITERATIONTARGET_TRANSLATION:
			AdvanceTranslation(objects[i], stepSize->tXY, stepSize->tZ);
			break;
		case ITERATIONTARGET_ROTATION:
			AdvanceRotation(objects[i], stepSize->r);
			break;
		}

		objects[i]->UpdateRendererFromPose(views[j]);
	}
}
void OptimisationEngine::AdvanceTranslation(Object3D* object, float stepSizeXY, float stepSizeZ)
{
	object->pose->translation->x -= stepSizeXY * object->dpose->translation->x;
	object->pose->translation->y -= stepSizeXY * object->dpose->translation->y;
	object->pose->translation->z -= stepSizeZ * object->dpose->translation->z;
}
void OptimisationEngine::AdvanceRotation(Object3D* object, float stepSizeR)
{
	object->pose->rotation->vector4d.x -= stepSizeR * object->dpose->rotation->vector4d.x;
	object->pose->rotation->vector4d.y -= stepSizeR * object->dpose->rotation->vector4d.y;
	object->pose->rotation->vector4d.z -= stepSizeR * object->dpose->rotation->vector4d.z;
	object->pose->rotation->vector4d.w -= stepSizeR * object->dpose->rotation->vector4d.w;
}

void OptimisationEngine::NormaliseRotation()
{
	int i, j;
	for (i = 0; i < objectCount; i++) 
	{
		objects[i]->pose->rotation->Normalize();
		for (j = 0; j < viewCount; j++) objects[i]->UpdateRendererFromPose(views[j]);
	}
}

bool OptimisationEngine::HasConverged()
{
	return false;
}

void OptimisationEngine::GetImage(PerseusImage* image, GetImageType getImageType, Object3D* object, View3D* view)
{
	switch (getImageType)
	{
	case GETIMAGE_WIREFRAME:
		DrawingEngine::Instance()->Draw(object, view, object->pose, object->imageWireframe[view->viewId], DrawingEngine::RENDERING_WIREFRAME);
		ImageUtils::Instance()->Copy(object->imageWireframe[view->viewId], image);
		break;
	case GETIMAGE_FILL:
		DrawingEngine::Instance()->Draw(object, view, object->pose, object->imageFill[view->viewId], DrawingEngine::RENDERING_FILL);
		ImageUtils::Instance()->Copy(object->imageFill[view->viewId], image);
		break;
	case GETIMAGE_ORIGINAL:
		ImageUtils::Instance()->Copy(view->imageRegistered, image);
		break;
	case GETIMAGE_POSTERIORS:
		break;
	case GETIMAGE_SIHLUETTE:
		ImageUtils::Instance()->Copy(view->imageRegistered, image);
		getProcessedDataDTSihluetteLSDXDY(object, view);
		ImageUtils::Instance()->Overlay(object->imageSihluette[view->viewId], image);
		break;
	case GETIMAGE_DT:
		break;
	case GETIMAGE_PROXIMITY_GL:
		break;
	case GETIMAGE_PROXIMITY:
		ImageUtils::Instance()->Copy(view->imageRegistered, image);
		//getProcessedDataDTSihluetteLSDXDY(object, view);
		//ImageUtils::Instance()->Overlay(object->imageSihluette[view->viewId], image);
		DrawingEngine::Instance()->Draw(object, view, object->pose, object->imageWireframe[view->viewId], DrawingEngine::RENDERING_WIREFRAME);
		ImageUtils::Instance()->Overlay(object->imageWireframe[view->viewId], image);
		break;
	}
}