#include "UndoSetParentRecord.h"

#include "Blueberry\Scene\Components\Transform.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Events\SceneEvents.h"

namespace Blueberry
{
	UndoSetParentRecord::UndoSetParentRecord(Transform* transform, Transform* newParent, bool worldPositionStays, size_t siblingIndex) : m_Transform(transform), m_BeforeParent(transform->GetParent()), m_AfterParent(newParent), m_BeforeSiblingIndex(transform->GetSiblingIndex()), m_AfterSiblingIndex(siblingIndex)
	{
		bool parentChanging = m_BeforeParent != m_AfterParent;
		bool siblingIndexChanging = m_BeforeSiblingIndex != m_AfterSiblingIndex;
		if (parentChanging)
		{
			Entity* entity = transform->GetEntity();
			if (PrefabManager::IsPrefabChild(entity))
			{
				PrefabManager::RemoveParent(entity);
			}
			if (PrefabManager::IsPartOfPrefabInstance(newParent))
			{
				PrefabManager::SetParent(entity, newParent);
			}
			transform->SetParent(newParent, worldPositionStays);
		}
		if (parentChanging || siblingIndexChanging)
		{
			transform->SetSiblingIndex(siblingIndex);
		}
		SceneEvents::GetHierarchyUpdated().Invoke();
	}

	void UndoSetParentRecord::OnUndo()
	{
		bool parentChanging = m_BeforeParent != m_AfterParent;
		bool siblingIndexChanging = m_BeforeSiblingIndex != m_AfterSiblingIndex;
		if (parentChanging)
		{
			Entity* entity = m_Transform->GetEntity();
			Transform* newParent = m_BeforeParent.Get();
			if (PrefabManager::IsPrefabChild(entity))
			{
				PrefabManager::RemoveParent(entity);
			}
			if (PrefabManager::IsPartOfPrefabInstance(newParent))
			{
				PrefabManager::SetParent(entity, newParent);
			}
			m_Transform->SetParent(m_BeforeParent.Get());
		}
		if (parentChanging || siblingIndexChanging)
		{
			m_Transform->SetSiblingIndex(m_BeforeSiblingIndex);
		}
		SceneEvents::GetHierarchyUpdated().Invoke();
	}

	void UndoSetParentRecord::OnRedo()
	{
		bool parentChanging = m_BeforeParent != m_AfterParent;
		bool siblingIndexChanging = m_BeforeSiblingIndex != m_AfterSiblingIndex;
		if (parentChanging)
		{
			Entity* entity = m_Transform->GetEntity();
			Transform* newParent = m_AfterParent.Get();
			if (PrefabManager::IsPrefabChild(entity))
			{
				PrefabManager::RemoveParent(entity);
			}
			if (PrefabManager::IsPartOfPrefabInstance(newParent))
			{
				PrefabManager::SetParent(entity, newParent);
			}
			m_Transform->SetParent(m_AfterParent.Get());
		}
		if (parentChanging || siblingIndexChanging)
		{
			m_Transform->SetSiblingIndex(m_AfterSiblingIndex);
		}
		SceneEvents::GetHierarchyUpdated().Invoke();
	}
}