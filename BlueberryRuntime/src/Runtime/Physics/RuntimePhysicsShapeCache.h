#pragma once

#include "Blueberry\Physics\PhysicsShapeCache.h"

namespace Blueberry
{
	class RuntimePhysicsShapeCache : public PhysicsShapeCache
	{
	protected:
		virtual void ClearImpl(Mesh* mesh) final;
		virtual bool TryLoadImpl(Mesh* mesh, bool isConvex, const Vector3& scale, List<uint8_t>& data) final;
		virtual void SaveImpl(Mesh* mesh, bool isConvex, const Vector3& scale, List<uint8_t>& data) final;
	};
}