#pragma once

#include "Collider.h"

namespace Blueberry
{
	class BB_API SphereCollider : public Collider
	{
		OBJECT_DECLARATION(SphereCollider)

	public:
		const float& GetRadius();

	private:
		virtual JPH::Shape* GetShape() override;

	private:
		float m_Radius = 1.0f;
	};
}