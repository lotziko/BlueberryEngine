#pragma once

#include "Collider.h"

namespace Blueberry
{
	class Mesh;

	class BB_API MeshCollider : public Collider
	{
		OBJECT_DECLARATION(MeshCollider)

	public:
		MeshCollider() = default;
		virtual ~MeshCollider() = default;

	private:
		virtual JPH::Shape* GetShape() override;

	private:
		ObjectPtr<Mesh> m_Mesh;
		bool m_IsConvex = false;
	};
}