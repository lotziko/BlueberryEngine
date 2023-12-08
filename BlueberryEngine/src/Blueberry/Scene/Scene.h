#pragma once

#include "EnityComponent.h"
#include "Components\ComponentManager.h"
#include <stack>

namespace Blueberry
{
	class Camera;
	class Serializer;

	class Scene
	{
	public:
		Scene();

		void Serialize(Serializer& serializer);
		void Deserialize(Serializer& serializer);

		bool Initialize();

		template<class ComponentType>
		ComponentIterator GetIterator();

		void Destroy();

		Ref<Entity> CreateEntity(const std::string& name);
		void AddEntity(Ref<Entity>& entity);
		void DestroyEntity(Entity* entity);
		void DestroyEntity(Ref<Entity>& entity);

		const std::vector<Ref<Entity>>& GetEntities();

	private:
		std::vector<Ref<Entity>> m_Entities;
		ComponentManager m_ComponentManager;

		std::stack<std::size_t> m_EmptyEntityIds;
		std::size_t m_MaxEntityId = 0;

		friend class Entity;
	};

	template<class ComponentType>
	inline ComponentIterator Scene::GetIterator()
	{
		return m_ComponentManager.GetIterator<ComponentType>();
	}
}