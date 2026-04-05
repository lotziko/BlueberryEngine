#include "RuntimePhysicsShapeCache.h"

namespace Blueberry
{
	void RuntimePhysicsShapeCache::ClearImpl(Mesh* mesh)
	{
	}

	bool RuntimePhysicsShapeCache::TryLoadImpl(Mesh* mesh, bool isConvex, const Vector3& scale, List<uint8_t>& data)
	{
		return false;
	}

	void RuntimePhysicsShapeCache::SaveImpl(Mesh* mesh, bool isConvex, const Vector3& scale, List<uint8_t>& data)
	{
	}
}