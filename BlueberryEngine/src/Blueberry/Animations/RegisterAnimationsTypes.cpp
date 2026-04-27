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
		REGISTER_DATA_CLASS(AnimationGraphConditionData);
		REGISTER_DATA_CLASS(AnimationGraphParameterData);
		REGISTER_CLASS(AnimationState);
		REGISTER_CLASS(AnimationTransition);
		REGISTER_CLASS(AnimationStateMachine);
		REGISTER_CLASS(AnimationGraph);
	}
}
