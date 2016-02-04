#ifndef __PERSEUS_POSE3D__
#define __PERSEUS_POSE3D__

#include "..\Renderer\Primitives\Quaternion.h"
#include "..\Primitives\Vector3D.h"

using namespace Perseus::Primitives;
using namespace Renderer::Primitives;

namespace Perseus
{
	namespace Objects
	{
		class Pose3D
		{
		public:
			Quaternion *rotation;
			VECTOR3DA *translation;

			void Set(float *pose) { this->Set(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6]); }

			void Set(Pose3D *pose) { 
				this->translation->x = pose->translation->x;
				this->translation->y = pose->translation->y;
				this->translation->z = pose->translation->z;
				this->rotation->vector4d.x = pose->rotation->vector4d.x;
				this->rotation->vector4d.y = pose->rotation->vector4d.y;
				this->rotation->vector4d.z = pose->rotation->vector4d.z;
				this->rotation->vector4d.w = pose->rotation->vector4d.w;				
			}

			void Set(VECTOR3DA* translation, Quaternion* rotation){
				rotation->CopyInto(this->rotation);
				this->translation->x = translation->x;
				this->translation->y = translation->y;
				this->translation->z = translation->z;
			}
			
			void Set(VFLOAT tX, VFLOAT tY, VFLOAT tZ, VFLOAT rX, VFLOAT rY, VFLOAT rZ) {
				this->translation->x = tX;
				this->translation->y = tY;
				this->translation->z = tZ;
				this->rotation->SetFromEuler(rX,rY,rZ);
			}	
			
			void Set(VFLOAT tX, VFLOAT tY, VFLOAT tZ, VFLOAT rX, VFLOAT rY, VFLOAT rZ, VFLOAT rW) {
				this->translation->x = tX;
				this->translation->y = tY;
				this->translation->z = tZ;
				this->rotation->vector4d.x = rX;
				this->rotation->vector4d.y = rY;
				this->rotation->vector4d.z = rZ;
				this->rotation->vector4d.w = rW;
			}	

			void Get(VECTOR3DA* translation, Quaternion* rotation) {
				this->rotation->CopyInto(rotation);
				translation->x = this->translation->x;
				translation->y = this->translation->y;
				translation->z = this->translation->z;
			}

			void CopyInto(Pose3D *targetPose) {
				targetPose->translation->x = this->translation->x;
				targetPose->translation->y = this->translation->y;
				targetPose->translation->z = this->translation->z;
				targetPose->rotation->vector4d.x = this->rotation->vector4d.x;
				targetPose->rotation->vector4d.y = this->rotation->vector4d.y;
				targetPose->rotation->vector4d.z = this->rotation->vector4d.z;
				targetPose->rotation->vector4d.w = this->rotation->vector4d.w;
			}

			void Clear()
			{
				this->translation->x = 0; this->translation->y = 0; this->translation->z = 0;
				this->rotation->vector4d.x = 0; this->rotation->vector4d.y = 0; this->rotation->vector4d.z = 0; this->rotation->vector4d.w = 0;
			}

			Pose3D(VFLOAT tX, VFLOAT tY, VFLOAT tZ, VFLOAT rX, VFLOAT rY, VFLOAT rZ) { 
				rotation = new Quaternion();  translation = new VECTOR3DA(); 
				this->Set(tX, tY, tZ, rX, rY, rZ);
			}

			Pose3D(void) { rotation = new Quaternion();  translation = new VECTOR3DA(); }
			~Pose3D(void) { if (rotation) delete rotation; if (translation) delete translation; }
		};
	}
}

#endif
