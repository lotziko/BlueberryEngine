#include "bbpch.h"
#include "Scene.h"

#include "Blueberry\Serialization\Serializer.h"

#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	Scene::Scene()
	{
	}

	void Scene::Serialize(Serializer& serializer)
	{
		for (auto entity : m_Entities)
		{
			serializer.Serialize(entity);
			for (auto component : entity->GetComponents())
			{
				serializer.Serialize(component);
			}
		}
	}

	void Scene::Deserialize(ryml::NodeRef& root)
	{
		//ryml::NodeRef entitiesNode = node["m_Entities"];
		//for (auto entityNode : entitiesNode)
		//{
		//	auto k = entityNode.key();// TODO store all objects as one list and deserialize them after creating
		//	bool b = entityNode.has_key_anchor();// Use tree root in node argument and remove tree from context
		//	auto anchor = entityNode.key_anchor();
		//	FileId fileId = std::stoull(std::string(anchor.data(), anchor.len));
		//	Ref<Entity> entity = CreateEntity("Entity");
		//	context.AddObject(fileId, entity->GetObjectId());
		//	entity->Deserialize(context, entityNode);
		//}
	}

	bool Scene::Initialize()
	{
		return true;
	}

	void Scene::Destroy()
	{
		for (auto entity : m_Entities)
		{
			DestroyEntity(entity);
		}
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