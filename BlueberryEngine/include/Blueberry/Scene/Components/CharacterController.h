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

		virtual void OnEnable() final;
		virtual void OnDisable() final;
		virtual void OnUpdate() final;

		const float& GetHeight();
		const float& GetRadius();

	private:
		ObjectPtr<Transform> m_CameraTransform;
		float m_Height = 2.0f;
		float m_Radius = 0.3f;

		struct PrivateData;

		Transform* m_Transform;
		PrivateData* m_PrivateData;
		alignas(8) char m_PrivateStorage[24];
		bool m_Initialized;
		Vector2 m_Rotation;
		Vector3 m_Velocity;
	};
}