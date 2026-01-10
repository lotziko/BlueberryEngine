#include "RegisterAnimationsTypes.h"

#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Animations\AnimationClip.h"
#include "Blueberry\Animations\AnimationGraph.h"

namespace Blueberry
{
	void RegisterAnimationsTypes()
	{
		REGISTER_DATA_CLASS(AnimationBoneData);
		REGISTER_CLASS(AnimationClip);
		REGISTER_CLASS(AnimationGraph);
	}
}
