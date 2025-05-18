#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Entity;
	class Transform;
	class Scene;

	class BB_API Component : public Object
	{
		OBJECT_DECLARATION(Component)

	public:
		virtual ~Component() = default;
		
		Entity* GetEntity();
		Transform* GetTransform();
		Scene* GetScene();
		
		virtual void OnBeginPlay() { };
		virtual void OnEnable() { };
		virtual void OnDisable() { };
		virtual void OnUpdate() { };

	protected:
		void AddToSceneComponents(const size_t& type);
		void RemoveFromSceneComponents(const size_t& type);

	protected:
		ObjectPtr<Entity> m_Entity;

		bool m_IsActive = false;

		friend class Entity;
	};

	class BB_API UpdatableComponent
	{
	public:
		static const size_t Type;
	};
}