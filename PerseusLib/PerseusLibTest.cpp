#include "PerseusLib.h"

#include "Utils/Timer.h"

int main(void)
{
	char str[100];
	int i;

	int width = 640, height = 480;
	int viewCount = 1, objectCount = 1;

	Timer t;

	PerseusImage* result = new PerseusImage(width, height, PerseusImage::IMAGE_RGBA, false, false, false);
	PerseusImage* camera = new PerseusImage(width, height, PerseusImage::IMAGE_RGBA, false, false, false);

	ImageUtils::Instance()->LoadImageFromFile(camera, "Files/Images/box.png");

	//Object3DParams *object3DParams = new Object3DParams(0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f); //Forceps
	Object3DParams *object3DParams = new Object3DParams(-1.28f, -2.90f, 37.47f, -40.90f, -207.77f, 27.48f); //Box
	//Object3DParams *object3DParams = new Object3DParams(-0.75f, 0.0f, 5.0f, 180, 0, 0); //Monkey
	View3DParams *view3DParams = new View3DParams();

	Object3D **objects = new Object3D*[objectCount];
	View3D **views = new View3D*[viewCount];

	objects[0] = new Object3D(0, viewCount, "Files/Models/Renderer/box.obj", width, height, object3DParams, view3DParams);
	views[0] = new View3D(0, "Files/CameraCalibration/900nc.cal", width, height, view3DParams);

	//Pose3D* histogramPose = new Pose3D(0.0f, 0.0f, 5.0f, -90.0f, 0.0f, 0.0f); //Forceps
	Pose3D* histogramPose = new Pose3D(-2.98f, -2.90f, 37.47f, -40.90f, -207.77f, 27.48f); //Box
	//Pose3D* histogramPose = new Pose3D(-0.75f, 0.0f, 5.0f, 180, 0, 0); //Monkey
	HistogramEngine::Instance()->UpdateVarBinHistogram(objects[0], views[0], camera, histogramPose);
	ImageUtils::Instance()->SaveImageToFile(objects[0]->imageHistogramMask[0],"Files/Images/mask.bmp");

	//DrawingEngine::Instance()->Draw(objects[0], views[0], histogramPose, objects[0]->imageHistogramMask[0], DrawingEngine::RENDERING_FILL);
	//FILE *f = fopen("c:/temp/out.txt","w+");
	//for (int i=0; i<640*480;i++)
	//	fprintf(f,"%f ",float(objects[0]->imageHistogramMask[0]->zbuffer[i])/float(MAX_INT));
	//fclose(f);

	IterationConfiguration *iterConfig = new IterationConfiguration();
	iterConfig->width = width;
	iterConfig->height = height;
	iterConfig->iterObjectsId[0] = 0;
	iterConfig->iterViewsId[0] = 0;
	iterConfig->objectCount = 1;
	iterConfig->viewCount = 1;
	iterConfig->levelSetBandSize = 30;
	iterConfig->stepSize = new StepSize3D(0.2f, 0.5f, 10.0f);
	iterConfig->iterCount = 1;

	OptimisationEngine::Instance()->Initialise(width, height);

	OptimisationEngine::Instance()->RegisterViewImage(views[0], camera);

	for (i=0; i<4; i++)
	{
		switch (i)
		{
		case 0: 
			iterConfig->useCUDAEF = true;
			iterConfig->useCUDARender = true;
			break;
		case 1: 
			iterConfig->useCUDAEF = true;
			iterConfig->useCUDARender = false;
			break;
		case 2: 
			iterConfig->useCUDAEF = false;
			iterConfig->useCUDARender = true;
			break;
		case 3: 
			iterConfig->useCUDAEF = false;
			iterConfig->useCUDARender = false;
			break;
		}
		sprintf(str, "Files/Images/result%04d.png", i);

		t.restart();
		OptimisationEngine::Instance()->Minimise(objects, views, iterConfig);
		t.check("Iteration");

		//objects[0]->pose->Set(0.0f, 0.0f, 5.0f, -90.0f, 0.0f, 0.0f);
		OptimisationEngine::Instance()->GetImage(result, GETIMAGE_PROXIMITY, objects[0], views[0]);

		ImageUtils::Instance()->SaveImageToFile(result, str);

		printf("%f %f %f %f %f %f %f\n", objects[0]->pose->translation->x, objects[0]->pose->translation->y, objects[0]->pose->translation->z,
			objects[0]->pose->rotation->vector4d.x, objects[0]->pose->rotation->vector4d.y, objects[0]->pose->rotation->vector4d.z,
			objects[0]->pose->rotation->vector4d.w);

		//iterConfig->initialPose->Set(objects[0]->pose);
	}

	OptimisationEngine::Instance()->Shutdown();

	delete object3DParams;
	delete view3DParams;

	for (i = 0; i<objectCount; i++) delete objects[i];
	delete objects;

	for (i = 0; i<viewCount; i++) delete views[i];
	delete views;

	delete result;

	delete histogramPose;

	return 0;
}