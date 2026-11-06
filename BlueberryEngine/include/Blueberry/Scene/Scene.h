#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\PointerCache.h"
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
	class LightingSettings;
	
	class BB_API Scene
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		Scene();
		~Scene();

		template<class ComponentType>
		ComponentView<ComponentType> GetComponents();

		ComponentView<Component> GetComponents(TypeId type);

		Scene* Clone();

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

		LightingSettings* GetLightingSettings() const;
		void SetLightingSettings(LightingSettings* lightingSettings);

		uint32_t GetIndex() const;

		static Scene* Get(uint32_t index);

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
		LightingSettings* m_LightingSettings = nullptr;
		uint32_t m_Index = 0;

		static PointerCache<Scene> s_PointerCache;

		friend class Entity;
		friend class Transform;
	};

	template<class ComponentType>
	inline ComponentView<ComponentType> Scene::GetComponents()
	{
		return m_ComponentManager.GetComponents<ComponentType>();
	}
}