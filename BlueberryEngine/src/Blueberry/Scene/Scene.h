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

		bool Initialize();

		template<class ComponentType>
		ComponentIterator GetIterator();

		void Update(const float& deltaTime);

		void Destroy();

		Entity* CreateEntity(const std::string& name);
		void AddEntity(Entity* entity);
		void DestroyEntity(Entity* entity);

		const std::map<ObjectId, ObjectPtr<Entity>>& GetEntities();

	private:
		std::map<ObjectId, ObjectPtr<Entity>> m_Entities;
		ComponentManager m_ComponentManager;

		friend class Entity;
	};

	template<class ComponentType>
	inline ComponentIterator Scene::GetIterator()
	{
		return m_ComponentManager.GetIterator<ComponentType>();
	}
}