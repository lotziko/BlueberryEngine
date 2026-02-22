#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class Transform;

	class BB_API CharacterController : public Component
	{
		OBJECT_DECLARATION(CharacterController)

	public:
		CharacterController();
		virtual ~CharacterController() = default;

		virtual void OnCreate() final;
		virtual void OnDestroy() final;
		virtual void OnEnable() final;
		virtual void OnDisable() final;
		virtual void OnFixedUpdate() final;
		virtual void OnUpdate() final;

		const float& GetHeight();
		const float& GetRadius();

		void SetVelocity(const Vector3& velocity);

	private:
		float m_Height = 2.0f;
		float m_Radius = 0.3f;

		struct PrivateData;

		PrivateData* m_PrivateData;
		alignas(8) char m_PrivateStorage[24];
		Transform* m_Transform;
		Vector3 m_Velocity;
	};
}