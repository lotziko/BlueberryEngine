#include "bbpch.h"
#include "Scene.h"

#include "Blueberry\Serialization\Serializer.h"
#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Scene\Components\PhysicsBody.h"
#include "Blueberry\Scene\Components\CharacterController.h"

namespace Blueberry
{
	Scene::Scene()
	{
	}

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
		for (auto pair : m_Entities)
		{
			if (pair.second.IsValid())
			{
				DestroyEntity(pair.second.Get());
			}
		}
		m_Entities.clear();
	}

	Entity* Scene::CreateEntity(const std::string& name = "Entity")
	{
		//BB_INFO("Entity is created.")
		Entity* entity = Object::Create<Entity>();
		entity->m_Scene = this;
		entity->m_Name = name;
		m_Entities[entity->GetObjectId()] = entity;

		entity->AddComponent<Transform>();
		entity->OnCreate();
		return entity;
	}

	void Scene::AddEntity(Entity* entity)
	{
		//BB_INFO("Entity is added.")
		entity->m_Scene = this;
		m_Entities[entity->GetObjectId()] = entity;
		if (entity->IsActiveInHierarchy())
		{
			entity->UpdateComponents();
		}
		for (auto& component : entity->GetComponents())
		{
			entity->AddToCreatedComponents(component);
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
		m_Entities.erase(entity->GetObjectId());
		Object::Destroy(entity);
	}

	const ska::flat_hash_map<ObjectId, ObjectPtr<Entity>>& Scene::GetEntities()
	{
		return m_Entities;
	}

	RendererTree& Scene::GetRendererTree()
	{
		return m_RendererTree;
	}
}