#pragma once

#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Core\ObjectPtr.h"

#include <Jolt\Jolt.h>
#include <Jolt\Physics\Body\BodyID.h>

namespace Blueberry
{
	class Transform;
	class Collider;

	enum class BodyType
	{
		Static,
		Kinematic,
		Dynamic
	};

	class PhysicsBody : public Component
	{
		OBJECT_DECLARATION(PhysicsBody)

	public:
		PhysicsBody() = default;
		~PhysicsBody() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;
		virtual void OnUpdate() final;

		static void BindProperties();

	private:
		BodyType m_BodyType;

		Transform* m_Transform;
		JPH::BodyID m_BodyId;
		bool m_IsInitialized = false;
		std::vector<ObjectPtr<Collider>> m_Colliders;

		friend class Collider;
	};
}