#pragma once

#include "EnityComponent.h"
#include "Components\ComponentManager.h"
#include <stack>

namespace Blueberry
{
	class Camera;
	struct ServiceContainer;

	class Scene
	{
	public:
		Scene();
		virtual ~Scene() = default;

		bool Initialize();

		template<class ComponentType>
		ComponentIterator GetIterator();

		Ref<Entity> CreateEntity(const std::string& name);
		void DestroyEntity(Ref<Entity>& entity);

		const std::vector<Ref<Entity>>& GetEntities() { return m_Entities; }

	protected:
		std::vector<Ref<Entity>> m_Entities;
		ComponentManager m_ComponentManager;

		std::stack<std::size_t> m_EmptyEntityIds;
		std::size_t m_MaxEntityId = 0;

		friend class Entity;
	};

	inline Ref<Entity> Scene::CreateEntity(const std::string& name = "Entity")
	{
		Ref<Entity> entity = CreateRef<Entity>(name);
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

	inline void Scene::DestroyEntity(Ref<Entity>& entity)
	{
		entity->Destroy();
		m_Entities[entity->m_Id] = nullptr;
		m_EmptyEntityIds.push(entity->m_Id);
		entity.reset();
	}

	template<class ComponentType>
	inline ComponentIterator Scene::GetIterator()
	{
		return m_ComponentManager.GetIterator<ComponentType>();
	}
}