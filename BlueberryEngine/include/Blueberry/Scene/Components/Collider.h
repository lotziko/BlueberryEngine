#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace JPH
{
	class Shape;
}

namespace Blueberry
{
	class PhysicsBody;

	class BB_API Collider : public Component
	{
		OBJECT_DECLARATION(Collider)

	public:
		Collider();
		virtual ~Collider() = default;

		virtual void OnCreate() final;
		virtual void OnDestroy() final;
		virtual void OnEnable() final;
		virtual void OnDisable() final;

		PhysicsBody* GetPhysicsBody();

	private:
		virtual JPH::Shape* GetShape() = 0;

	private:
		ObjectPtr<PhysicsBody> m_PhysicsBody;
		
		struct PrivateData;

		PrivateData* m_PrivateData;
		alignas(8) char m_PrivateStorage[4];

		friend class PhysicsBody;
	};
}