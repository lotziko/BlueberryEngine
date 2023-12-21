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
			serializer.AddObject(entity.Get());
			for (auto component : entity->GetComponents())
			{
				serializer.AddObject(component);
			}
		}
		serializer.Serialize(path);
	}

	void Scene::Deserialize(Serializer& serializer, const std::string& path)
	{
		serializer.Deserialize(path);
		for (auto& object : serializer.GetDeserializedObjects())
		{
			if (object->IsClassType(Entity::Type))
			{
				AddEntity((Entity*)object);
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
		Entity* entity = Object::Create<Entity>();
		entity->m_Scene = this;
		entity->m_Name = name;

		entity->AddComponent<Transform>();
		entity->m_Transform = WeakObjectPtr<Transform>(entity->GetComponent<Transform>());

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

		return entity;
	}

	void Scene::AddEntity(Entity* entity)
	{
		entity->m_Scene = this;
		entity->m_Transform = entity->GetComponent<Transform>();
		entity->m_Id = m_MaxEntityId;
		++m_MaxEntityId;
		m_Entities.emplace_back(entity);
	}

	void Scene::DestroyEntity(Entity* entity)
	{
		entity->Destroy();
		m_Entities[entity->m_Id] = nullptr;
		m_EmptyEntityIds.push(entity->m_Id);
		Object::Destroy(entity);
	}

	const std::vector<WeakObjectPtr<Entity>>& Scene::GetEntities()
	{
		return m_Entities;
	}
}