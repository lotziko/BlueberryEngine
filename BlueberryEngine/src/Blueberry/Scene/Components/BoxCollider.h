#pragma once

#include "Collider.h"

namespace Blueberry
{
	class BoxCollider : public Collider
	{
		OBJECT_DECLARATION(BoxCollider)

	public:
		static void BindProperties();

	private:
		virtual JPH::Shape* GetShape() override;
	};
}