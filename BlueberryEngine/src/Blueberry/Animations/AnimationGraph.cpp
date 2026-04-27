#include "Blueberry\Animations\AnimationGraph.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Animations\AnimationClip.h"

namespace Blueberry
{
	DATA_DEFINITION(AnimationGraphConditionData)
	{
		DEFINE_FIELD(AnimationGraphConditionData, m_Name, BindingType::String, FieldOptions())
		DEFINE_FIELD(AnimationGraphConditionData, m_Comparison, BindingType::Enum, FieldOptions())
		DEFINE_FIELD(AnimationGraphConditionData, m_Value, BindingType::Float, FieldOptions())
	}

	DATA_DEFINITION(AnimationGraphParameterData)
	{
		DEFINE_FIELD(AnimationGraphParameterData, m_Name, BindingType::String, FieldOptions())
		DEFINE_FIELD(AnimationGraphParameterData, m_Type, BindingType::Enum, FieldOptions())
		DEFINE_FIELD(AnimationGraphParameterData, m_Value, BindingType::Float, FieldOptions())
	}

	OBJECT_DEFINITION(AnimationState, Object)
	{
		DEFINE_FIELD(AnimationState, m_Name, BindingType::String, FieldOptions())
		DEFINE_FIELD(AnimationState, m_Position, BindingType::Vector2, FieldOptions().SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(AnimationState, m_Transitions, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&AnimationTransition::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(AnimationState, m_AnimationClip, BindingType::ObjectPtr, FieldOptions().SetObjectType(&AnimationClip::Type))
		DEFINE_FIELD(AnimationState, m_Speed, BindingType::Float, FieldOptions())
	}

	OBJECT_DEFINITION(AnimationTransition, Object)
	{
		DEFINE_FIELD(AnimationTransition, m_Conditions, BindingType::DataList, FieldOptions().SetObjectType(&AnimationGraphConditionData::Type))
		DEFINE_FIELD(AnimationTransition, m_Destination, BindingType::ObjectPtr, FieldOptions().SetObjectType(&AnimationState::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(AnimationTransition, m_IsFixedDuration, BindingType::Bool, FieldOptions())
		DEFINE_FIELD(AnimationTransition, m_TransitionOffset, BindingType::Float, FieldOptions())
		DEFINE_FIELD(AnimationTransition, m_TransitionDuration, BindingType::Float, FieldOptions())
		DEFINE_FIELD(AnimationTransition, m_HasExitTime, BindingType::Bool, FieldOptions())
		DEFINE_FIELD(AnimationTransition, m_ExitTime, BindingType::Float, FieldOptions())
	}

	OBJECT_DEFINITION(AnimationStateMachine, Object)
	{
		DEFINE_FIELD(AnimationStateMachine, m_States, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&AnimationState::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(AnimationStateMachine, m_DefaultState, BindingType::ObjectPtr, FieldOptions().SetObjectType(&AnimationState::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(AnimationStateMachine, m_EntryStatePosition, BindingType::Vector2, FieldOptions().SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(AnimationStateMachine, m_AnyStateTransitions, BindingType::ObjectPtrList, FieldOptions().SetObjectType(&AnimationTransition::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(AnimationStateMachine, m_AnyStatePosition, BindingType::Vector2, FieldOptions().SetVisibility(VisibilityType::Hidden))
	}

	OBJECT_DEFINITION(AnimationGraph, Object)
	{
		DEFINE_FIELD(AnimationGraph, m_StateMachine, BindingType::ObjectPtr, FieldOptions().SetObjectType(&AnimationStateMachine::Type).SetVisibility(VisibilityType::Hidden))
		DEFINE_FIELD(AnimationGraph, m_Parameters, BindingType::DataList, FieldOptions().SetObjectType(&AnimationGraphParameterData::Type))
	}

	void AnimationGraphConditionData::Initialize(AnimationParameterType type)
	{
		m_NameHash = TO_HASH(m_Name);
		m_Type = type;
	}

	const String& AnimationGraphConditionData::GetName() const
	{
		return m_Name;
	}

	void AnimationGraphConditionData::SetName(const String& name)
	{
		m_Name = name;
		m_NameHash = 0;
	}

	size_t AnimationGraphConditionData::GetNameHash() const
	{
		return m_NameHash;
	}

	AnimationParameterType AnimationGraphConditionData::GetType() const
	{
		return m_Type;
	}

	AnimationConditionComparison AnimationGraphConditionData::GetComparison() const
	{
		return m_Comparison;
	}

	bool AnimationGraphConditionData::GetBoolValue() const
	{
		return m_Value > 0.0f ? true : false;
	}

	int AnimationGraphConditionData::GetIntValue() const
	{
		return static_cast<int32_t>(m_Value);
	}

	float AnimationGraphConditionData::GetFloatValue() const
	{
		return m_Value;
	}

	const String& AnimationGraphParameterData::GetName() const
	{
		return m_Name;
	}

	void AnimationGraphParameterData::SetName(const String& name)
	{
		m_Name = name;
	}

	AnimationParameterType AnimationGraphParameterData::GetType() const
	{
		return m_Type;
	}

	bool AnimationGraphParameterData::GetBoolValue() const
	{
		return m_Value > 0;
	}

	void AnimationGraphParameterData::SetBoolValue(bool value)
	{
		m_Type = AnimationParameterType::Bool;
		m_Value = value ? 1.0f : 0.0f;
	}

	bool AnimationGraphParameterData::GetTriggerValue() const
	{
		return m_Value > 0;
	}

	void AnimationGraphParameterData::SetTriggerValue(bool value)
	{
		m_Type = AnimationParameterType::Trigger;
		m_Value = value ? 1.0f : 0.0f;
	}

	int AnimationGraphParameterData::GetIntValue() const
	{
		return static_cast<int>(m_Value);
	}

	void AnimationGraphParameterData::SetIntValue(int value)
	{
		m_Type = AnimationParameterType::Int;
		m_Value = static_cast<float>(value);
	}

	float AnimationGraphParameterData::GetFloatValue() const
	{
		return m_Value;
	}

	void AnimationGraphParameterData::SetFloatValue(float value)
	{
		m_Type = AnimationParameterType::Float;
		m_Value = value;
	}

	const Vector2& AnimationState::GetPosition() const
	{
		return m_Position;
	}

	void AnimationState::SetPosition(const Vector2& position)
	{
		m_Position = position;
	}

	const List<ObjectPtr<AnimationTransition>>& AnimationState::GetTransitions()
	{
		return m_Transitions;
	}

	AnimationTransition* AnimationState::CreateTransition(AnimationState* destination)
	{
		AnimationTransition* transition = Object::Create<AnimationTransition>();
		transition->SetDestination(destination);
		m_Transitions.push_back(transition);
		return transition;
	}

	void AnimationState::RemoveTransition(AnimationTransition* transition)
	{
		for (size_t i = 0; i < m_Transitions.size(); ++i)
		{
			if (m_Transitions[i] == transition)
			{
				m_Transitions.erase(m_Transitions.begin() + i);
				return;
			}
		}
	}

	uint32_t AnimationState::GetBoneIndex(const String& name) const
	{
		if (m_AnimationClip.IsValid())
		{
			return m_AnimationClip->GetBoneIndex(name);
		}
		return UINT32_MAX;
	}

	TRS AnimationState::GetTRS(float time, size_t index)
	{
		if (m_AnimationClip.IsValid())
		{
			return m_AnimationClip->GetTRS(time, index);
		}
		return {};
	}

	AnimationClip* AnimationState::GetClip()
	{
		return m_AnimationClip.Get();
	}

	float AnimationState::GetSpeed() const
	{
		return m_Speed;
	}

	AnimationState* AnimationStateMachine::CreateState()
	{
		AnimationState* state = Object::Create<AnimationState>();
		state->SetName("State");
		m_States.push_back(state);
		return state;
	}

	void AnimationStateMachine::RemoveState(AnimationState* state)
	{
		for (size_t i = 0; i < m_States.size(); ++i)
		{
			if (m_States[i] == state)
			{
				m_States.erase(m_States.begin() + i);
				return;
			}
		}
	}

	AnimationState* AnimationStateMachine::GetDefaultState() const
	{
		return m_DefaultState.Get();
	}

	void AnimationStateMachine::SetDefaultState(AnimationState* defaultState)
	{
		m_DefaultState = defaultState;
	}

	List<ObjectPtr<AnimationState>>& AnimationStateMachine::GetStates()
	{
		return m_States;
	}

	void AnimationGraph::InitializeIfNeeded()
	{
		if (m_IsInitialized)
		{
			return;
		}
		if (m_StateMachine.IsValid())
		{
			m_IsInitialized = true;
			for (auto& state : m_StateMachine->GetStates())
			{
				if (!state.IsValid())
				{
					continue;
				}
				for (auto& transition : state->GetTransitions())
				{
					if (!transition.IsValid())
					{
						continue;
					}
					for (auto& condition : transition.Get()->GetConditions())
					{
						for (auto& parameter : m_Parameters)
						{
							if (condition.GetName() == parameter.GetName())
							{
								condition.Initialize(parameter.GetType());
							}
						}
					}
				}
			}
			for (auto& transition : m_StateMachine->GetAnyStateTransitions())
			{
				if (!transition.IsValid())
				{
					continue;
				}
				for (auto& condition : transition.Get()->GetConditions())
				{
					for (auto& parameter : m_Parameters)
					{
						if (condition.GetName() == parameter.GetName())
						{
							condition.Initialize(parameter.GetType());
						}
					}
				}
			}
		}
	}

	AnimationStateMachine* AnimationGraph::GetStateMachine()
	{
		if (!m_StateMachine.IsValid())
		{
			m_StateMachine = Object::Create<AnimationStateMachine>();
		}
		return m_StateMachine.Get();
	}

	const Vector2& AnimationStateMachine::GetEntryStatePosition() const
	{
		return m_EntryStatePosition;
	}

	void AnimationStateMachine::SetEntryStatePosition(const Vector2& entryStatePosition)
	{
		m_EntryStatePosition = entryStatePosition;
	}

	AnimationTransition* AnimationStateMachine::CreateAnyStateTransition(AnimationState* destination)
	{
		AnimationTransition* transition = Object::Create<AnimationTransition>();
		transition->SetDestination(destination);
		m_AnyStateTransitions.push_back(transition);
		return transition;
	}

	void AnimationStateMachine::RemoveAnyStateTransition(AnimationTransition* transition)
	{
		for (size_t i = 0; i < m_AnyStateTransitions.size(); ++i)
		{
			if (m_AnyStateTransitions[i] == transition)
			{
				m_AnyStateTransitions.erase(m_AnyStateTransitions.begin() + i);
				return;
			}
		}
	}

	List<ObjectPtr<AnimationTransition>>& AnimationStateMachine::GetAnyStateTransitions()
	{
		return m_AnyStateTransitions;
	}

	const Vector2& AnimationStateMachine::GetAnyStatePosition() const
	{
		return m_AnyStatePosition;
	}

	void AnimationStateMachine::SetAnyStatePosition(const Vector2& anyStatePosition)
	{
		m_AnyStatePosition = anyStatePosition;
	}

	List<AnimationGraphParameterData>& AnimationGraph::GetParameters()
	{
		return m_Parameters;
	}

	List<AnimationGraphConditionData>& AnimationTransition::GetConditions()
	{
		return m_Conditions;
	}

	AnimationState* AnimationTransition::GetDestination()
	{
		return m_Destination.Get();
	}

	void AnimationTransition::SetDestination(AnimationState* destination)
	{
		m_Destination = destination;
	}

	bool AnimationTransition::IsFixedDuration() const
	{
		return m_IsFixedDuration;
	}

	float AnimationTransition::GetTransitionOffset() const
	{
		return m_TransitionOffset;
	}

	float AnimationTransition::GetTransitionDuration() const
	{
		return m_TransitionDuration;
	}

	bool AnimationTransition::HasExitTime() const
	{
		return m_HasExitTime;
	}

	float AnimationTransition::GetExitTime() const
	{
		return m_ExitTime;
	}
}