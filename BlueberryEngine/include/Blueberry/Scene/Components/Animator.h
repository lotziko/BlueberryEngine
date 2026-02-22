#pragma once

#include "Component.h"

namespace Blueberry
{
	class AnimationGraph;
	class Transform;

	class Animator : public Component
	{
		OBJECT_DECLARATION(Animator)

	public:
		Animator() = default;
		virtual ~Animator() = default;

		virtual void OnEnable() final;
		virtual void OnUpdate() final;
		
		AnimationGraph* GetAnimationGraph();
		void SetAnimationGraph(AnimationGraph* animationGraph);

	private:
		void RefreshState();

	private:
		ObjectPtr<AnimationGraph> m_AnimationGraph;
		List<std::pair<uint32_t, ObjectPtr<Transform>>> m_Bones;
		float m_Time = 0.0f;
	};
}