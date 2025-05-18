#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace JPH
{
	class Shape;
}

namespace Blueberry
{
	class BB_API Collider : public Component
	{
		OBJECT_DECLARATION(Collider)

	public:
		Collider() = default;
		virtual ~Collider() = default;

	private:
		virtual JPH::Shape* GetShape() = 0;

		virtual void OnBeginPlay() final;

		friend class PhysicsBody;
	};
}