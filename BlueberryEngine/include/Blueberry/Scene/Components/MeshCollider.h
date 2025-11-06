#pragma once

#include "Collider.h"

namespace Blueberry
{
	class Mesh;

	class BB_API MeshCollider : public Collider
	{
		OBJECT_DECLARATION(MeshCollider)

	public:
		const float& GetRadius();

	private:
		virtual JPH::Shape* GetShape() override;

	private:
		ObjectPtr<Mesh> m_Mesh;
	};
}