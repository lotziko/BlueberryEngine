#pragma once

#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Transform;
	class Collider;

	enum class BB_API BodyType
	{
		Static,
		Kinematic,
		Dynamic
	};

	class BB_API PhysicsBody : public Component
	{
		OBJECT_DECLARATION(PhysicsBody)

	public:
		PhysicsBody();
		virtual ~PhysicsBody() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;
		virtual void OnUpdate() final;

	private:
		BodyType m_BodyType;

		struct PrivateData;

		Transform* m_Transform;
		PrivateData* m_PrivateData;
		alignas(8) char m_PrivateStorage[4];
		bool m_IsInitialized = false;
		List<ObjectPtr<Collider>> m_Colliders;

		friend class Collider;
	};
}