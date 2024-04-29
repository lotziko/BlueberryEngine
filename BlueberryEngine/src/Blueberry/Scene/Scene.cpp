#include "bbpch.h"
#include "Scene.h"

#include "Blueberry\Serialization\Serializer.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	Scene::Scene()
	{
	}

	void Scene::Serialize(Serializer& serializer, const std::string& path)
	{
		for (auto& entity : m_Entities)
		{
			// Components are being added automatically
			serializer.AddObject(entity.Get());
		}
		serializer.Serialize(path);
	}

	void Scene::Deserialize(Serializer& serializer, const std::string& path)
	{
		serializer.Deserialize(path);
		for (auto& object : serializer.GetDeserializedObjects())
		{
			if (object.first->IsClassType(Entity::Type))
			{
				Entity* entity = (Entity*)object.first;
				AddEntity(entity);
			}
		}
	}

	bool Scene::Initialize()
	{
		return true;
	}

	void Scene::Destroy()
	{
		for (auto entity : m_Entities)
		{
			if (entity.IsValid())
			{
				DestroyEntity(entity.Get());
			}
		}
	}

	Entity* Scene::CreateEntity(const std::string& name = "Entity")
	{
		BB_INFO("Entity is created.")
		Entity* entity = Object::Create<Entity>();
		entity->m_Scene = this;
		entity->m_Name = name;

		if (m_EmptyEntityIds.size() > 0)
		{
			std::size_t id = m_EmptyEntityIds.top();
			entity->m_Id = id;
			m_EmptyEntityIds.pop();
			m_Entities[id] = entity;
		}
		else
		{
			entity->m_Id = m_MaxEntityId;
			++m_MaxEntityId;
			m_Entities.emplace_back(entity);
		}

		entity->AddComponent<Transform>();
		entity->m_Transform = ObjectPtr<Transform>(entity->GetComponent<Transform>());

		return entity;
	}

	void Scene::AddEntity(Entity* entity)
	{
		BB_INFO("Entity is added.")
		entity->m_Scene = this;
		entity->m_Transform = entity->GetComponent<Transform>();
		entity->m_Id = m_MaxEntityId;
		++m_MaxEntityId;
		m_Entities.emplace_back(entity);
		for (auto& component : entity->GetComponents())
		{
			entity->AddComponentIntoScene(component);
		}
		for (auto& child : entity->m_Transform->GetChildren())
		{
			AddEntity(child->GetEntity());
		}
	}

	void Scene::DestroyEntity(Entity* entity)
	{
		BB_INFO("Entity is destroyed.")
		entity->OnDestroy();
		m_Entities[entity->m_Id] = nullptr;
		m_EmptyEntityIds.push(entity->m_Id);
		Object::Destroy(entity);
	}

	const std::vector<ObjectPtr<Entity>>& Scene::GetEntities()
	{
		return m_Entities;
	}
}