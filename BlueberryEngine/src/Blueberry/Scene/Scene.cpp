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

	void Scene::Update(const float& deltaTime)
	{
		if (m_CreatedComponents.size() > 0)
		{
			for (auto& component : m_CreatedComponents)
			{
				component->OnBeginPlay();
			}
			m_CreatedComponents.clear();
		}

		// Update
		{
			for (auto& component : m_ComponentManager.GetIterator(UpdatableComponent::Type))
			{
				component.second->OnUpdate();
			}
		}
	}

	void Scene::Destroy()
	{
		for (auto& rootEntity : m_RootEntities)
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
		//BB_INFO("Entity is created.")
		Entity* entity = Object::Create<Entity>();
		entity->m_Scene = this;
		entity->m_Name = name;
		m_Entities[entity->GetObjectId()] = entity;
		m_RootEntities.emplace_back(entity);

		entity->AddComponent<Transform>();
		entity->OnCreate();
		return entity;
	}

	void Scene::AddEntity(Entity* entity)
	{
		//BB_INFO("Entity is added.")
		entity->m_Scene = this;
		m_Entities[entity->GetObjectId()] = entity;
		if (entity->GetTransform()->GetParent() == nullptr)
		{
			m_RootEntities.emplace_back(entity);
		}
		if (entity->IsActiveInHierarchy())
		{
			entity->UpdateComponents();
		}
		for (uint32_t i = 0; i < entity->GetComponentCount(); ++i)
		{
			entity->AddToCreatedComponents(entity->GetComponent(i));
		}
		for (auto& child : entity->GetTransform()->GetChildren())
		{
			AddEntity(child->GetEntity());
		}
	}

	void Scene::DestroyEntity(Entity* entity)
	{
		for (auto& child : entity->GetTransform()->GetChildren())
		{
			DestroyEntity(child->GetEntity());
		}
		//BB_INFO("Entity is destroyed.");
		entity->OnDestroy();
		if (entity->GetTransform()->GetParent() == nullptr)
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
		m_Entities.erase(entity->GetObjectId());
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
}