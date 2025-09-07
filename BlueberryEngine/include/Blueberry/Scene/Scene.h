#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Components\ComponentManager.h"
#include "Blueberry\Events\Event.h"
#include "Blueberry\Graphics\RendererTree.h"

namespace Blueberry
{
	class Camera;
	class Serializer;
	class Entity;
	class Component;
	
	class Scene
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Scene() = default;

		bool Initialize();

		template<class ComponentType>
		ComponentIterator GetIterator();

		template<class ComponentType>
		ComponentMap& GetComponents();

		void Update(const float& deltaTime);

		void Destroy();

		Entity* CreateEntity(const String& name);
		void AddEntity(Entity* entity);
		void DestroyEntity(Entity* entity);

		const Dictionary<ObjectId, ObjectPtr<Entity>>& GetEntities();
		const List<ObjectPtr<Entity>>& GetRootEntities();

		RendererTree& GetRendererTree();

	private:
		Dictionary<ObjectId, ObjectPtr<Entity>> m_Entities;
		List<ObjectPtr<Entity>> m_RootEntities;
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