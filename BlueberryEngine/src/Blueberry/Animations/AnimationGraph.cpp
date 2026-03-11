#include "Blueberry\Animations\AnimationGraph.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Animations\AnimationClip.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AnimationGraph, Object)
	{
		DEFINE_BASE_FIELDS(AnimationGraph, Object)
		DEFINE_FIELD(AnimationGraph, m_AnimationClip, BindingType::ObjectPtr, FieldOptions().SetObjectType(&AnimationClip::Type))
	}

	uint32_t AnimationGraph::GetBoneIndex(const String& name)
	{
		if (m_AnimationClip.IsValid())
		{
			return m_AnimationClip->GetBoneIndex(name);
		}
		return UINT32_MAX;
	}

	TRS AnimationGraph::GetTRS(const float& time, const size_t& index)
	{
		return m_AnimationClip->GetTRS(time, index);
	}
}