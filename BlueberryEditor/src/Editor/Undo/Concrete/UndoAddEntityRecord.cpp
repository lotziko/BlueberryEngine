#include "UndoAddEntityRecord.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Scene.h"
#include "Editor\EditorSceneManager.h"
#include "Editor\Misc\EntityHelper.h"
#include "Editor\Events\SceneEvents.h"

namespace Blueberry
{
	UndoAddEntityRecord::UndoAddEntityRecord(Entity* entity) : m_Entity(entity)
	{
		Scene* scene = EditorSceneManager::GetScene();
		m_SceneIndex = scene->GetIndex();
		scene->AddEntity(entity);
		Entity::Poll();
		SceneEvents::GetEntityCreated().Invoke();
	}

	void UndoAddEntityRecord::OnUndo()
	{
		Scene* scene = Scene::Get(m_SceneIndex);
		if (scene != nullptr)
		{
			Entity* entity = m_Entity.Get();
			if (m_Serializer.IsEmpty())
			{
				entity->GetTransform()->SetParent(nullptr);
				List<Object*> objects;
				EntityHelper::GatherObjects(entity, objects);
				m_Serializer.Serialize(objects);
			}
			scene->DestroyEntity(entity);	// TODO check PrefabInstance
			Entity::Poll();
			SceneEvents::GetEntityDestroyed().Invoke();
		}
	}

	void UndoAddEntityRecord::OnRedo()
	{
		Scene* scene = Scene::Get(m_SceneIndex);
		if (scene != nullptr)
		{
			m_Serializer.Deserialize();
			Entity* entity = m_Entity.Get();
			scene->AddEntity(entity);
			Entity::Poll();
			SceneEvents::GetEntityCreated().Invoke();
		}
	}
}