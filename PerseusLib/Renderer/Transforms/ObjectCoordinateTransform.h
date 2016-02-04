#ifndef __RENDERER_OBJECT_COORDINATE_TRANSFORM__
#define __RENDERER_OBJECT_COORDINATE_TRANSFORM__

#include "..\..\Primitives\Vector3D.h"

#include "..\..\Others\PerseusDefines.h"

#include "..\..\Renderer\Primitives\Quaternion.h"

using namespace Renderer::Primitives;

namespace Renderer
{
	namespace Transforms
	{
		class ObjectCoordinateTransform
		{
		public:
			VFLOAT *modelViewMatrix;

			Quaternion *rotation;
			VECTOR3DA translation;

			void SetTranslation(VFLOAT* translation) {
				this->translation.x = translation[0]; this->translation.y = translation[1]; this->translation.z = translation[2];
			}
			void SetTranslation(VECTOR3DA translation) { this->translation = translation; }
			void SetTranslation(VECTOR3DA* translation) { this->translation = *translation; }
		
			void AddTranslation(VFLOAT *translation) { 
				this->translation.x += translation[0]; this->translation.y += translation[1]; this->translation.z += translation[2];
			}
			void AddTranslation(VECTOR3DA translation) {
				this->translation.x += translation.x; this->translation.y += translation.y; this->translation.z += translation.z;
			}
			void AddTranslation(VECTOR3DA *translation) {
				this->translation.x += translation->x; this->translation.y += translation->y; this->translation.z += translation->z;
			}

			void SetRotation(Quaternion* rotation) { this->rotation = rotation; }

			void GetModelViewMatrix(VFLOAT* );

			ObjectCoordinateTransform(void);
			~ObjectCoordinateTransform(void);
		};
	}
}

#endif