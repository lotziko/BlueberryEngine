#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Variant.h"

namespace Blueberry
{
	class AnimationClip;
	class AnimationTransition;

	enum class AnimationParameterType
	{
		None,
		Bool,
		Trigger,
		Int,
		Float,
	};

	enum class AnimationConditionComparison
	{
		Greater,
		Less,
		Equal,
		NotEqual
	};

	class BB_API AnimationGraphConditionData : public Data
	{
		DATA_DECLARATION(AnimationGraphConditionData)
		
	public:
		void Initialize(AnimationParameterType type);

		const String& GetName() const;
		void SetName(const String& name);
		size_t GetNameHash() const;
		AnimationParameterType GetType() const;

		AnimationConditionComparison GetComparison() const;

		bool GetBoolValue() const;
		int GetIntValue() const;
		float GetFloatValue() const;
		
	private:
		String m_Name;
		AnimationConditionComparison m_Comparison = AnimationConditionComparison::Greater;
		float m_Value = 0;

		size_t m_NameHash = 0;
		AnimationParameterType m_Type = AnimationParameterType::None;
	};

	class BB_API AnimationGraphParameterData : public Data
	{
		DATA_DECLARATION(AnimationGraphParameterData)
		
	public:
		AnimationGraphParameterData() = default;

		const String& GetName() const;
		void SetName(const String& name);

		AnimationParameterType GetType() const;

		bool GetBoolValue() const;
		void SetBoolValue(bool value);

		bool GetTriggerValue() const;
		void SetTriggerValue(bool value);

		int GetIntValue() const;
		void SetIntValue(int value);

		float GetFloatValue() const;
		void SetFloatValue(float value);

	private:
		String m_Name;
		AnimationParameterType m_Type = AnimationParameterType::None;
		float m_Value = 0;
	};

	class BB_API AnimationState : public Object
	{
		OBJECT_DECLARATION(AnimationState)

	public:
		AnimationState() = default;
		virtual ~AnimationState() = default;

		const Vector2& GetPosition() const;
		void SetPosition(const Vector2& position);

		const List<ObjectPtr<AnimationTransition>>& GetTransitions();
		AnimationTransition* CreateTransition(AnimationState* destination);
		void RemoveTransition(AnimationTransition* transition);

		uint32_t GetBoneIndex(const String& name) const;
		TRS GetTRS(float time, size_t index);
		AnimationClip* GetClip();
		float GetSpeed() const;

	private:
		Vector2 m_Position;
		List<ObjectPtr<AnimationTransition>> m_Transitions;
		ObjectPtr<AnimationClip> m_AnimationClip;
		float m_Speed = 1.0f;
	};

	class BB_API AnimationTransition : public Object
	{
		OBJECT_DECLARATION(AnimationTransition)

	public:
		AnimationTransition() = default;
		virtual ~AnimationTransition() = default;

		List<AnimationGraphConditionData>& GetConditions();

		AnimationState* GetDestination();
		void SetDestination(AnimationState* destination);

		bool IsFixedDuration() const;
		float GetTransitionOffset() const;
		float GetTransitionDuration() const;
		bool HasExitTime() const;
		float GetExitTime() const;

	private:
		List<AnimationGraphConditionData> m_Conditions;
		ObjectPtr<AnimationState> m_Destination;
		bool m_IsFixedDuration = false;
		float m_TransitionOffset = 0.0f;
		float m_TransitionDuration = 0.0f;
		bool m_HasExitTime = false;
		float m_ExitTime = 0.0f;
	};

	class BB_API AnimationStateMachine : public Object
	{
		OBJECT_DECLARATION(AnimationStateMachine)

		AnimationState* CreateState();
		void RemoveState(AnimationState* state);

		AnimationState* GetDefaultState() const;
		void SetDefaultState(AnimationState* defaultState);

		List<ObjectPtr<AnimationState>>& GetStates();

		const Vector2& GetEntryStatePosition() const;
		void SetEntryStatePosition(const Vector2& entryStatePosition);

		AnimationTransition* CreateAnyStateTransition(AnimationState* destination);
		void RemoveAnyStateTransition(AnimationTransition* transition);

		List<ObjectPtr<AnimationTransition>>& GetAnyStateTransitions();

		const Vector2& GetAnyStatePosition() const;
		void SetAnyStatePosition(const Vector2& anyStatePosition);

	private:
		List<ObjectPtr<AnimationState>> m_States;
		ObjectPtr<AnimationState> m_DefaultState;
		Vector2 m_EntryStatePosition;
		List<ObjectPtr<AnimationTransition>> m_AnyStateTransitions;
		Vector2 m_AnyStatePosition;
	};

	class BB_API AnimationGraph : public Object
	{
		OBJECT_DECLARATION(AnimationGraph)

	public:
		AnimationGraph() = default;
		virtual ~AnimationGraph() = default;

		void InitializeIfNeeded();

		AnimationStateMachine* GetStateMachine();
		List<AnimationGraphParameterData>& GetParameters();

	private:
		ObjectPtr<AnimationStateMachine> m_StateMachine;
		List<AnimationGraphParameterData> m_Parameters;
		bool m_IsInitialized = false;
	};
}