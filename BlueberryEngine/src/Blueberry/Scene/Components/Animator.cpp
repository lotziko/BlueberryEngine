#include "Blueberry\Scene\Components\Animator.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Core\Time.h"
#include "Blueberry\Animations\AnimationGraph.h"
#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"

namespace Blueberry
{
	OBJECT_DEFINITION(Animator, Component)
	{
		DEFINE_BASE_FIELDS(Animator, Component)
		DEFINE_FIELD(Animator, m_AnimationGraph, BindingType::ObjectPtr, FieldOptions().SetObjectType(&AnimationGraph::Type).SetUpdateCallback(MethodBind::Create(&Animator::RefreshState)))
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

	void Animator::OnEnable()
	{
		RefreshState();
	}

	void Animator::OnUpdate()
	{
		if (m_AnimationGraph.IsValid())
		{
			m_Time += Time::GetDeltaTime();
			for (auto& bone : m_Bones)
			{
				bone.second->SetLocalTRS(m_AnimationGraph->GetTRS(m_Time, bone.first));
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
		RefreshState();
	}

	void Animator::RefreshState()
	{
		if (m_AnimationGraph.IsValid())
		{
			m_Time = 0.0f;
			m_Bones.clear();
			Dictionary<String, Transform*> bones;
			GatherBones(bones, GetTransform());
			for (auto& pair : bones)
			{
				uint32_t index = m_AnimationGraph->GetBoneIndex(pair.first);
				if (index != UINT32_MAX)
				{
					m_Bones.push_back(std::make_pair(index, pair.second));
				}
			}
		}
	}
}