#pragma once

#include "Collider.h"

namespace Blueberry
{
	class BB_API BoxCollider : public Collider
	{
		OBJECT_DECLARATION(BoxCollider)

	public:
		const Vector3& GetSize();

	private:
		virtual JPH::Shape* GetShape() override;

	private:
		Vector3 m_Size = Vector3::One;
	};
}