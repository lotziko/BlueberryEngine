#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Components\ComponentManager.h"
#include "Blueberry\Events\Event.h"
#include "Blueberry\Graphics\Octree.h"

namespace Blueberry
{
	class Camera;
	class Serializer;
	class Entity;
	class Component;
	class CullableInterface;
	
	class BB_API Scene
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Scene() = default;

		bool Initialize();

		template<class ComponentType>
		ComponentView<ComponentType> GetComponents();

		ComponentView<Component> GetComponents(TypeId type);

		void FixedUpdate();
		void Update();

		void Destroy();

		Entity* CreateEntity(const String& name);
		void AddEntity(Entity* entity);
		void RemoveEntity(Entity* entity);
		void DestroyEntity(Entity* entity);

		const Dictionary<ObjectId, ObjectPtr<Entity>>& GetEntities();
		const List<ObjectPtr<Entity>>& GetRootEntities();

		Octree& GetRendererTree();

		void MarkCullableDirty(ObjectId id, CullableInterface* object);
		bool FlushDirtyCullables();

	private:
		void AddToRoot(Entity* entity);
		void RemoveFromRoot(Entity* entity);
		const size_t GetRootIndex(Entity* entity);
		void SetRootIndex(Entity* entity, size_t index);

		void AddChildEntity(Entity* entity);

	private:
		Dictionary<ObjectId, ObjectPtr<Entity>> m_Entities;
		List<ObjectPtr<Entity>> m_RootEntities;

		// Stores only components with iterators
		ComponentManager m_ComponentManager;
		Octree m_RendererTree = Octree(Vector3::Zero, 10.0f, 1.0f, 1.0f);
		Dictionary<ObjectId, CullableInterface*> m_DirtyCullables;
		size_t m_CullingFrame = 0;

		friend class Entity;
		friend class Transform;
	};

	template<class ComponentType>
	inline ComponentView<ComponentType> Scene::GetComponents()
	{
		return m_ComponentManager.GetComponents<ComponentType>();
	}
}