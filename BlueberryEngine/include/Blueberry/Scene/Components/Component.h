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
		
		virtual const String& GetName() override;

		Entity* GetEntity();
		Transform* GetTransform();
		Scene* GetScene();

		const bool& IsActive();
		bool CanExecute();
		
		virtual void OnCreate() { };
		virtual void OnDestroy() { };
		virtual void OnEnable() { };
		virtual void OnDisable() { };
		virtual void OnFixedUpdate() { };
		virtual void OnUpdate() { };

	protected:
		ObjectPtr<Entity> m_Entity;

		bool m_IsCreated = false;
		bool m_IsActive = false;

		friend class Entity;
	};

	class BB_API UpdatableComponent
	{
	public:
		static const size_t Type;
	};
}