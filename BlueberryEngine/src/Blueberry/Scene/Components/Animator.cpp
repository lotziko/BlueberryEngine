#include "Blueberry\Scene\Components\Animator.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Time.h"
#include "Blueberry\Animations\AnimationGraph.h"
#include "Blueberry\Animations\AnimationClip.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Animator, Component)
	{
		DEFINE_BASE_FIELDS(Animator, Component)
		DEFINE_FIELD(Animator, m_AnimationGraph, BindingType::ObjectPtr, FieldOptions().SetObjectType(&AnimationGraph::Type).SetUpdateCallback(MethodBind::Create(&Animator::Initialize)))
		DEFINE_ITERATOR(UpdatableComponent)
	}

	void GatherBones(Dictionary<String, Transform*>& bones, Transform* parent)
	{
		Entity* entity = parent->GetEntity();
		if (entity == nullptr)
		{
			return;
		}
		bones.insert_or_assign(entity->GetName().c_str(), parent);
		if (parent->GetChildrenCount() > 0)
		{
			for (auto& child : parent->GetChildren())
			{
				GatherBones(bones, child.Get());
			}
		}
	}

	void Animator::OnCreate()
	{
		Initialize();
	}

	void Animator::OnUpdate()
	{
		if (m_AnimationGraph.IsValid())
		{
			if (m_ActiveStates[0].state == nullptr)
			{
				return;
			}

			// Update time
			float deltaTime = Time::GetDeltaTime();
			uint32_t stateCount = m_CurrentTransition == nullptr ? 1 : 2;
			for (uint32_t i = 0; i < stateCount; ++i)
			{
				StateData& stateData = m_ActiveStates[i];
				if (stateData.length > 0.0f)
				{
					stateData.previousNormalizedTime = stateData.normalizedTime;
					if (stateData.isLoop)
					{
						stateData.time = std::clamp(stateData.time + deltaTime * stateData.speed, 0.0f, stateData.length);
						stateData.normalizedTime = stateData.time / stateData.length;
						stateData.previousLoopNormalizedTime = stateData.loopNormalizedTime;
						stateData.loopNormalizedTime = std::fmodf(stateData.normalizedTime, stateData.length);
					}
					else
					{
						stateData.time += deltaTime * stateData.speed;
						stateData.normalizedTime = stateData.time / stateData.length;
					}
				}
				else
				{
					stateData.time = 0.0f;
					stateData.previousNormalizedTime = 0.0f;
					stateData.normalizedTime = 0.0f;
				}
			}
			if (m_CurrentTransition != nullptr)
			{
				m_CurrentTransitionTime += deltaTime * m_ActiveStates[0].speed;
			}
			
			// Find valid transitions
			if (m_IsDirty)
			{
				EvaluateTransitions();
				m_IsDirty = false;
			}

			// Assign or update transition
			if (m_CurrentTransition == nullptr)
			{
				StateData& currentState = m_ActiveStates[0];
				for (auto& pair : m_ValidTransitions)
				{
					bool isValid = false;
					if (currentState.length > 0.0f)
					{
						float exitTime = pair.second;
						if (exitTime <= 1.0f)
						{
							if (exitTime == 0.0f)
							{
								isValid = true;
							}
							else
							{
								if (currentState.isLoop)
								{
									float loopTime = currentState.loopNormalizedTime;
									if (loopTime < currentState.previousLoopNormalizedTime)
									{
										loopTime += 1.0f;
									}
									if (loopTime >= exitTime && currentState.previousLoopNormalizedTime < exitTime)
									{
										isValid = true;
									}
								}
								else
								{
									if (currentState.normalizedTime >= exitTime && currentState.previousNormalizedTime < exitTime)
									{
										isValid = true;
									}
								}
							}
						}
						else
						{
							if (currentState.normalizedTime >= exitTime && currentState.previousNormalizedTime < exitTime)
							{
								isValid = true;
							}
						}
					}
					else
					{
						isValid = true;
					}

					if (isValid)
					{
						AnimationTransition* transition = pair.first;
						AnimationState* destination = transition->GetDestination();
						float duration = transition->IsFixedDuration() ? transition->GetTransitionDuration() : (transition->GetTransitionDuration() * m_ActiveStates[0].length);
						float toTime = transition->GetTransitionOffset() * destination->GetClip()->GetLength();
						if (duration == 0.0f)
						{
							m_CurrentTransition = nullptr;
							ResetTriggers(transition);
							InitializeState(destination, 0);
							m_IsDirty = true;
						}
						else
						{
							m_CurrentTransition = pair.first;
							m_CurrentTransitionTime = 0.0f;
							m_CurrentTransitionDuration = duration;
							InitializeState(destination, 1);
							m_ActiveStates[1].time = toTime;
						}
						break;
					}
				}
			}
			else
			{
				if (m_CurrentTransitionTime >= m_CurrentTransitionDuration)
				{
					ResetTriggers(m_CurrentTransition);
					m_CurrentTransition = nullptr;
					m_ActiveStates[0] = m_ActiveStates[1];
					m_ActiveStates[1].state = nullptr;
					m_IsDirty = true;
				}
			}

			// Update pose
			if (m_CurrentTransition == nullptr)
			{
				StateData& stateData = m_ActiveStates[0];
				for (auto& bone : m_Bones)
				{
					bone.second->SetLocalTRS(stateData.clip->GetTRS(stateData.time, bone.first));
				}
			}
			else
			{
				float t = std::clamp(m_CurrentTransitionTime / m_CurrentTransitionDuration, 0.0f, 1.0f);
				StateData& firstStateData = m_ActiveStates[0];
				StateData& secondStateData = m_ActiveStates[1];
				for (auto& bone : m_Bones)
				{
					TRS fromState = firstStateData.state->GetTRS(firstStateData.time, bone.first);
					TRS toState = secondStateData.state->GetTRS(secondStateData.time, bone.first);
					bone.second->SetLocalTRS({ Vector3::Lerp(fromState.position, toState.position, t), Quaternion::Slerp(fromState.rotation, toState.rotation, t), Vector3::Lerp(fromState.scale, toState.scale, t) });
				}
			}
		}
	}
	
	AnimationGraph* Animator::GetAnimationGraph()
	{
		return m_AnimationGraph.Get();
	}

	void Animator::SetAnimationGraph(AnimationGraph* animationGraph)
	{
		m_AnimationGraph = animationGraph;
		Initialize();
	}

	void Animator::SetBool(size_t first, bool value)
	{
		float newValue = value ? 1.0f : 0.0f;
		for (auto& pair : m_Values)
		{
			if (pair.first == first)
			{
				pair.second = newValue;
				m_IsDirty = true;
				return;
			}
		}
	}

	void Animator::SetTrigger(size_t first)
	{
		for (auto& pair : m_Values)
		{
			if (pair.first == first)
			{
				pair.second = 1.0f;
				m_IsDirty = true;
				return;
			}
		}
	}

	void Animator::SetInt(size_t first, int32_t value)
	{
		float newValue = static_cast<float>(value);
		for (auto& pair : m_Values)
		{
			if (pair.first == first)
			{
				pair.second = newValue;
				m_IsDirty = true;
				return;
			}
		}
	}

	void Animator::SetFloat(size_t first, float value)
	{
		for (auto& pair : m_Values)
		{
			if (pair.first == first)
			{
				pair.second = value;
				m_IsDirty = true;
				return;
			}
		}
	}

	void Animator::Initialize()
	{
		if (m_AnimationGraph.IsValid())
		{
			m_AnimationGraph->InitializeIfNeeded();
			m_ActiveStates[0] = {};
			m_ActiveStates[1] = {};
			m_Bones.clear();
			AnimationStateMachine* stateMachine = m_AnimationGraph->GetStateMachine();
			if (stateMachine != nullptr)
			{
				InitializeState(stateMachine->GetDefaultState(), 0);
				if (m_ActiveStates[0].state != nullptr)
				{
					m_Values.clear();
					m_IsDirty = true;
					auto& parameters = m_AnimationGraph->GetParameters();
					for (size_t i = 0; i < parameters.size(); ++i)
					{
						auto& parameter = parameters[i];
						m_Values.push_back(std::make_pair(TO_HASH(parameter.GetName()), parameter.GetFloatValue()));
					}
					Dictionary<String, Transform*> bones;
					GatherBones(bones, GetTransform());
					for (auto& pair : bones)
					{
						uint32_t index = m_ActiveStates[0].state->GetBoneIndex(pair.first);
						if (index != UINT32_MAX)
						{
							m_Bones.push_back(std::make_pair(index, pair.second));
						}
					}
				}
			}
		}
	}

	void Animator::EvaluateTransition(AnimationTransition* transition)
	{
		if (transition == nullptr)
		{
			return;
		}
		bool isValid = true;
		float exitTime = 0.0f;
		if (transition->HasExitTime())
		{
			exitTime = transition->GetExitTime();
		}
		for (auto& condition : transition->GetConditions())
		{
			size_t first = condition.GetNameHash();
			for (auto& pair : m_Values)
			{
				if (first == pair.first)
				{
					float value = condition.GetFloatValue();
					switch (condition.GetType())
					{
					case AnimationParameterType::Bool:
						if (value != pair.second)
						{
							isValid = false;
						}
						break;
					case AnimationParameterType::Trigger:
						if (!pair.second)
						{
							isValid = false;
						}
						break;
					case AnimationParameterType::Int:
						switch (condition.GetComparison())
						{
						case AnimationConditionComparison::Greater:
							if (value > pair.second)
							{
								isValid = false;
							}
							break;
						case AnimationConditionComparison::Less:
							if (value < pair.second)
							{
								isValid = false;
							}
							break;
						case AnimationConditionComparison::Equal:
							if (value == pair.second)
							{
								isValid = false;
							}
							break;
						case AnimationConditionComparison::NotEqual:
							if (value != pair.second)
							{
								isValid = false;
							}
							break;
						}
						break;
					case AnimationParameterType::Float:
						switch (condition.GetComparison())
						{
						case AnimationConditionComparison::Greater:
							if (value > pair.second)
							{
								isValid = false;
							}
							break;
						case AnimationConditionComparison::Less:
							if (value < pair.second)
							{
								isValid = false;
							}
							break;
						}
						break;
					}
					if (!isValid)
					{
						break;
					}
				}
			}
			if (!isValid)
			{
				break;
			}
		}
		if (isValid)
		{
			m_ValidTransitions.push_back(std::make_pair(transition, exitTime));
		}
	}

	void Animator::EvaluateTransitions()
	{
		m_ValidTransitions.clear();
		auto& parameters = m_AnimationGraph->GetParameters();
		for (auto& transition : m_ActiveStates[0].state->GetTransitions())
		{
			EvaluateTransition(transition.Get());
		}
		for (auto& transition : m_AnimationGraph->GetStateMachine()->GetAnyStateTransitions())
		{
			EvaluateTransition(transition.Get());
		}
	}

	void Animator::InitializeState(AnimationState* state, uint32_t index)
	{
		AnimationClip* clip = state->GetClip();
		if (clip == nullptr)
		{
			m_ActiveStates[index].state = nullptr;
			return;
		}
		StateData stateData = {};
		stateData.state = state;
		stateData.clip = clip;
		stateData.isLoop = clip->IsLoop();
		stateData.length = clip->GetLength();
		stateData.speed = state->GetSpeed();
		m_ActiveStates[index] = stateData;
	}

	void Animator::ResetTriggers(AnimationTransition* transition)
	{
		for (auto& condition : transition->GetConditions())
		{
			size_t first = condition.GetNameHash();
			for (auto& pair : m_Values)
			{
				if (condition.GetType() == AnimationParameterType::Trigger && first == pair.first)
				{
					pair.second = 0.0f;
					break;
				}
			}
		}
	}
}