#include "ObjectCoordinateTransform.h"

using namespace Renderer::Transforms;

ObjectCoordinateTransform::ObjectCoordinateTransform(void)
{
	modelViewMatrix = new VFLOAT[16];

	for (int i=0; i<16; i++)
		modelViewMatrix[i] = 0;

	translation = VECTOR3DA(0,0,0);
}

ObjectCoordinateTransform::~ObjectCoordinateTransform(void)
{
	delete modelViewMatrix;
}

void ObjectCoordinateTransform::GetModelViewMatrix(VFLOAT *returnMatrix)
{
	int i;

	returnMatrix[0] = 1;
	returnMatrix[5] = 1;
	returnMatrix[10] = 1;
	returnMatrix[15] = 1;
	returnMatrix[3] = 0;
	returnMatrix[7] = 0;
	returnMatrix[11] = 0;

	returnMatrix[12] = this->translation.x;
	returnMatrix[13] = this->translation.y;
	returnMatrix[14] = this->translation.z;

	VFLOAT matrixFromSource[16]; 
	rotation->GetMatrix(matrixFromSource);

	returnMatrix[0] = matrixFromSource[0];
	returnMatrix[1] = matrixFromSource[1];
	returnMatrix[2] = matrixFromSource[2];
	returnMatrix[4] = matrixFromSource[4];
	returnMatrix[5] = matrixFromSource[5];
	returnMatrix[6] = matrixFromSource[6];
	returnMatrix[8] = matrixFromSource[8];
	returnMatrix[9] = matrixFromSource[9];
	returnMatrix[10] = matrixFromSource[10];

	VFLOAT norm = 1.0f/returnMatrix[15];
	for (i=0; i<16; i++) returnMatrix[i] *= norm;
}