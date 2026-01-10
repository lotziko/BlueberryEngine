#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class AnimationClip;

	class BB_API AnimationGraph : public Object
	{
		OBJECT_DECLARATION(AnimationGraph)

	public:
		AnimationGraph() = default;
		virtual ~AnimationGraph() = default;

		uint32_t GetBoneIndex(const String& name);
		TRS GetTRS(const float& time, const size_t& index);

	private:
		ObjectPtr<AnimationClip> m_AnimationClip;
	};
}