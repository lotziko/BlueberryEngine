#include "UndoDestroyEntityRecord.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Scene.h"
#include "Editor\Prefabs\PrefabManager.h"
#include "Editor\Prefabs\PrefabInstance.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Misc\EntityHelper.h"
#include "Editor\Events\SceneEvents.h"

namespace Blueberry
{
	UndoDestroyEntityRecord::UndoDestroyEntityRecord(Entity* entity) : m_Entity(entity)
	{
		Transform* transform = entity->GetTransform();
		m_Parent = transform->GetParent();
		m_SiblingIndex = transform->GetSiblingIndex();
		transform->SetParent(nullptr);
		List<Object*> objects;
		EntityHelper::GatherObjects(entity, objects);
		for (Object* object : objects)
		{
			if (object->GetType() == Entity::Type)
			{
				if (PrefabManager::IsPrefabInstanceRoot(entity))
				{
					m_PrefabInstances.push_back(PrefabManager::GetInstance(entity));
				}
			}
		}
		m_Serializer.Serialize(objects);
		EditorSceneManager::GetScene()->DestroyEntity(entity);
		Entity::Poll();
		SceneEvents::GetEntityDestroyed().Invoke();
	}

	UndoDestroyEntityRecord::~UndoDestroyEntityRecord()
	{
		for (auto& prefabInstance : m_PrefabInstances)
		{
			Object::Destroy(prefabInstance.Get());
		}
	}

	void UndoDestroyEntityRecord::OnUndo()
	{
		m_Serializer.Deserialize();
		Entity* entity = m_Entity.Get();
		Transform* transform = entity->GetTransform();
		EditorSceneManager::GetScene()->AddEntity(entity);
		transform->SetParent(m_Parent.Get());
		transform->SetSiblingIndex(m_SiblingIndex);
		Entity::Poll();
		SceneEvents::GetEntityCreated().Invoke();
	}

	void UndoDestroyEntityRecord::OnRedo()
	{
		EditorSceneManager::GetScene()->DestroyEntity(m_Entity.Get());
		Entity::Poll();
		SceneEvents::GetEntityDestroyed().Invoke();
	}
}