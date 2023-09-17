#include "bbpch.h"
#include "Scene.h"

namespace Blueberry
{
	Scene::Scene()
	{
	}

	Scene::~Scene()
	{
		for (auto entity : m_Entities)
		{
			DestroyEntity(entity);
		}
	}

	bool Scene::Initialize()
	{
		return true;
	}

	Ref<Entity> Scene::CreateEntity(const std::string& name = "Entity")
	{
		Ref<Entity> entity = ObjectDB::CreateObject<Entity>(name);
		entity->m_Scene = this;

		entity->AddComponent<Transform>();
		entity->m_Transform = entity->GetComponent<Transform>();

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

	void Scene::DestroyEntity(Entity* entity)
	{
		entity->Destroy();
		m_Entities[entity->m_Id] = nullptr;
		m_EmptyEntityIds.push(entity->m_Id);
		ObjectDB::DestroyObject(entity);
	}

	void Scene::DestroyEntity(Ref<Entity>& entity)
	{
		DestroyEntity(entity.get());
		entity.reset();
	}

	const std::vector<Ref<Entity>>& Scene::GetEntities()
	{
		return m_Entities;
	}
}