#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace JPH
{
	class Shape;
}

namespace Blueberry
{
	class Collider : public Component
	{
		OBJECT_DECLARATION(Collider)

	private:
		virtual JPH::Shape* GetShape() = 0;

		friend class PhysicsBody;
	};
}