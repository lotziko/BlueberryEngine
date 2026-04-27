#pragma once

#include "Blueberry\Scene\Components\Component.h"
#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Transform;
	class Collider;

	class BB_API PhysicsBody : public Component
	{
		OBJECT_DECLARATION(PhysicsBody)

	public:
		PhysicsBody();
		virtual ~PhysicsBody() = default;

		virtual void OnCreate() final;
		virtual void OnDestroy() final;
		virtual void OnEnable() final;
		virtual void OnDisable() final;
		virtual void OnFixedUpdate() final;
		virtual void OnUpdate() final;

		void Move(const Vector3& position);
		void Move(const Quaternion& rotation);
		void Move(const Vector3& position, const Quaternion& rotation);

		void AddImpulse(const Vector3& impulse);
		void AddImpulse(const Vector3& impulse, const Vector3& position);

	private:
		bool m_IsKinematic = false;

		struct PrivateData;

		Transform* m_Transform;
		PrivateData* m_PrivateData;
		alignas(8) char m_PrivateStorage[4];
		List<ObjectPtr<Collider>> m_Colliders;
		size_t m_UpdateCount;

		friend class Collider;
	};
}