#pragma once

#include "Collider.h"

namespace Blueberry
{
	class SphereCollider : public Collider
	{
		OBJECT_DECLARATION(SphereCollider)

	public:
		static void BindProperties();

	private:
		virtual JPH::Shape* GetShape() override;

	private:
		float m_Radius = 1.0f;
	};
}