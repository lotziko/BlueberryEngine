#include "Blueberry\Scene\Scene.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Serialization\Serializer.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	bool Scene::Initialize()
	{
		return true;
	}

	void Scene::FixedUpdate()
	{
		for (auto& component : m_ComponentManager.GetIterator(UpdatableComponent::Type))
		{
			component.second->OnFixedUpdate();
		}
	}

	void Scene::Update()
	{
		for (auto& component : m_ComponentManager.GetIterator(UpdatableComponent::Type))
		{
			component.second->OnUpdate();
		}
	}

	void Scene::Destroy()
	{
		auto snapshot = m_RootEntities;
		for (auto& rootEntity : snapshot)
		{
			if (rootEntity.IsValid())
			{
				DestroyEntity(rootEntity.Get());
			}
		}
		m_Entities.clear();
		m_RootEntities.clear();
	}

	Entity* Scene::CreateEntity(const String& name = "Entity")
	{
		Entity* entity = Object::Create<Entity>();
		entity->m_Scene = this;
		entity->m_Name = name;
		m_Entities[entity->GetObjectId()] = entity;
		AddToRoot(entity);

		entity->AddComponent<Transform>();
		entity->UpdateHierarchy();
		return entity;
	}

	void Scene::AddEntity(Entity* entity)
	{
		if (entity == nullptr)
		{
			return;
		}
		if (entity->m_Scene != this)
		{
			entity->m_Scene = this;
			m_Entities[entity->GetObjectId()] = entity;
			if (entity->GetTransform()->GetParent() == nullptr)
			{
				m_RootEntities.push_back(entity);
			}
		}
		for (auto& child : entity->GetTransform()->GetChildren())
		{
			AddChildEntity(child.Get()->GetEntity());
		}
		entity->UpdateHierarchy();
	}

	void Scene::RemoveEntity(Entity* entity)
	{
		if (entity == nullptr)
		{
			return;
		}
		entity->DisableComponents();
		entity->m_Scene = nullptr;
		if (entity->GetTransform()->GetParent() == nullptr)
		{
			RemoveFromRoot(entity);
		}
		m_Entities.erase(entity->GetObjectId());
		for (auto& child : entity->GetTransform()->GetChildren())
		{
			RemoveEntity(child.Get()->GetEntity());
		}
	}

	void Scene::DestroyEntity(Entity* entity)
	{
		auto snapshot = entity->GetTransform()->GetChildren();
		for (auto& child : snapshot)
		{
			DestroyEntity(child.Get()->GetEntity());
		}
		if (entity->GetTransform()->GetParent() == nullptr)
		{
			RemoveFromRoot(entity);
		}
		m_Entities.erase(entity->GetObjectId());
		entity->OnDestroy();
		Object::Destroy(entity);
	}

	const Dictionary<ObjectId, ObjectPtr<Entity>>& Scene::GetEntities()
	{
		return m_Entities;
	}

	const List<ObjectPtr<Entity>>& Scene::GetRootEntities()
	{
		return m_RootEntities;
	}

	RendererTree& Scene::GetRendererTree()
	{
		return m_RendererTree;
	}

	void Scene::AddToRoot(Entity* entity)
	{
		m_RootEntities.push_back(entity);
	}

	void Scene::RemoveFromRoot(Entity* entity)
	{
		for (auto it = m_RootEntities.begin(); it < m_RootEntities.end(); ++it)
		{
			if (it->Get() == entity)
			{
				m_RootEntities.erase(it);
				break;
			}
		}
	}

	const size_t Scene::GetRootIndex(Entity* entity)
	{
		for (size_t i = 0; i < m_RootEntities.size(); ++i)
		{
			if (m_RootEntities[i].Get() == entity)
			{
				return i;
			}
		}
		return 0;
	}

	void Scene::SetRootIndex(Entity* entity, size_t index)
	{
		size_t oldIndex = 0;
		for (size_t i = 0; i < m_RootEntities.size(); ++i)
		{
			if (m_RootEntities[i].Get() == entity)
			{
				oldIndex = i;
				break;
			}
		}
		m_RootEntities.move_element(oldIndex, index);
	}

	void Scene::AddChildEntity(Entity* entity)
	{
		if (entity->m_Scene != this)
		{
			entity->m_Scene = this;
			m_Entities[entity->GetObjectId()] = entity;
		}

		for (auto& child : entity->GetTransform()->GetChildren())
		{
			AddChildEntity(child.Get()->GetEntity());
		}
	}
}