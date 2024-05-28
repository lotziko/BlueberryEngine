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
		for (auto& pair : m_Entities)
		{
			// Components are being added automatically
			serializer.AddObject(pair.second.Get());
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
		ObjectPtr<Entity> entity;
		while(m_Entities.size() > 0 && (entity = m_Entities.begin()->second).IsValid())
		{
			DestroyEntity(entity.Get());
		}
	}

	Entity* Scene::CreateEntity(const std::string& name = "Entity")
	{
		//BB_INFO("Entity is created.")
		Entity* entity = Object::Create<Entity>();
		entity->m_Scene = this;
		entity->m_Name = name;
		m_Entities[entity->GetObjectId()] = entity;

		entity->AddComponent<Transform>();
		entity->m_Transform = ObjectPtr<Transform>(entity->GetComponent<Transform>());

		return entity;
	}

	void Scene::AddEntity(Entity* entity)
	{
		//BB_INFO("Entity is added.")
		entity->m_Scene = this;
		entity->m_Transform = entity->GetComponent<Transform>();
		m_Entities[entity->GetObjectId()] = entity;

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
		for (auto& child : entity->GetTransform()->GetChildren())
		{
			DestroyEntity(child->GetEntity());
		}
		//BB_INFO("Entity is destroyed.");
		entity->OnDestroy();
		m_Entities.erase(entity->GetObjectId());
		Object::Destroy(entity);
	}

	const std::map<ObjectId, ObjectPtr<Entity>>& Scene::GetEntities()
	{
		return m_Entities;
	}
}