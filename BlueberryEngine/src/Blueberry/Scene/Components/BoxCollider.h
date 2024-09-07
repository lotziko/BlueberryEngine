#pragma once

#include "Collider.h"

namespace Blueberry
{
	class BoxCollider : public Collider
	{
		OBJECT_DECLARATION(BoxCollider)

	public:
		const Vector3& GetSize();

		static void BindProperties();

	private:
		virtual JPH::Shape* GetShape() override;

	private:
		Vector3 m_Size = Vector3::One;
	};
}