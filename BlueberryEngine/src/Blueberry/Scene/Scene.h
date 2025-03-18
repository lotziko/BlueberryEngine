#pragma once

#include "EnityComponent.h"
#include "Components\ComponentManager.h"
#include "Blueberry\Events\Event.h"
#include "Blueberry\Graphics\RendererTree.h"

namespace Blueberry
{
	class Camera;
	class Serializer;

	class Scene
	{
	public:
		BB_OVERRIDE_NEW_DELETE;

		Scene();

		bool Initialize();

		template<class ComponentType>
		ComponentIterator GetIterator();

		template<class ComponentType>
		ComponentMap& GetComponents();

		void Update(const float& deltaTime);

		void Destroy();

		Entity* CreateEntity(const std::string& name);
		void AddEntity(Entity* entity);
		void DestroyEntity(Entity* entity);

		const ska::flat_hash_map<ObjectId, ObjectPtr<Entity>>& GetEntities();

		RendererTree& GetRendererTree();

	private:
		ska::flat_hash_map<ObjectId, ObjectPtr<Entity>> m_Entities;
		List<Component*> m_CreatedComponents;
		// Stores only components added using AddToSceneComponents()
		ComponentManager m_ComponentManager;
		RendererTree m_RendererTree;

		friend class Entity;
	};

	template<class ComponentType>
	inline ComponentIterator Scene::GetIterator()
	{
		return m_ComponentManager.GetIterator<ComponentType>();
	}

	template<class ComponentType>
	inline ComponentMap& Scene::GetComponents()
	{
		return m_ComponentManager.GetComponents(ComponentType::Type);
	}
}