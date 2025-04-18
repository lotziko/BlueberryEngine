#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class Transform;

	class CharacterController : public Component
	{
		OBJECT_DECLARATION(CharacterController)

	public:
		CharacterController() = default;
		virtual ~CharacterController();

		virtual void OnEnable() final;
		virtual void OnDisable() final;
		virtual void OnUpdate() final;

		const float& GetHeight();
		const float& GetRadius();

	private:
		float m_Height = 2.0f;
		float m_Radius = 0.3f;

		struct CharacterData;

		Transform* m_Transform;
		CharacterData* m_CharacterData;
	};
}