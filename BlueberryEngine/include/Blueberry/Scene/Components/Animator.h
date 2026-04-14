#pragma once

#include "Component.h"

namespace Blueberry
{
	class AnimationGraph;
	class AnimationState;
	class AnimationClip;
	class AnimationTransition;
	class Transform;

	class BB_API Animator : public Component
	{
		OBJECT_DECLARATION(Animator)

	public:
		Animator() = default;
		virtual ~Animator() = default;

		virtual void OnEnable() final;
		virtual void OnUpdate() final;
		
		AnimationGraph* GetAnimationGraph();
		void SetAnimationGraph(AnimationGraph* animationGraph);

		void SetBool(size_t nameHash, bool value);
		void SetTrigger(size_t nameHash);
		void SetInt(size_t nameHash, int32_t value);
		void SetFloat(size_t nameHash, float value);

	private:
		void Initialize();
		void EvaluateTransition(AnimationTransition* transition);
		void EvaluateTransitions();
		void InitializeState(AnimationState* state, uint32_t index);
		void ResetTriggers(AnimationTransition* transition);

	private:
		ObjectPtr<AnimationGraph> m_AnimationGraph;
		List<std::pair<uint32_t, ObjectPtr<Transform>>> m_Bones;

		List<std::pair<size_t, float>> m_Values;
		List<std::pair<AnimationTransition*, float>> m_ValidTransitions;
		bool m_IsDirty = true;

		struct StateData
		{
			AnimationState* state = nullptr;
			AnimationClip* clip = nullptr;
			bool isLoop = false;
			float length = 0.0f;
			float speed = 0.0f;
			float time = 0.0f;
			float normalizedTime = 0.0f;
			float previousNormalizedTime = 0.0f;
			float loopNormalizedTime = 0.0f;
			float previousLoopNormalizedTime = 0.0f;
		};

		StateData m_ActiveStates[2];

		AnimationTransition* m_CurrentTransition = nullptr;
		float m_CurrentTransitionTime = 0.0f;
		float m_CurrentTransitionDuration = 0.0f;
	};
}