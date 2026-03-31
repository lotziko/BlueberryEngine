#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class Mesh;

	class PhysicsShapeCache
	{
	public:
		static void Initialize(PhysicsShapeCache* shapeCache);
		static void Clear(Mesh* mesh);
		static void* GetShape(Mesh* mesh, bool isConvex, const Vector3& scale);

	protected:
		virtual void ClearImpl(Mesh* mesh) = 0;
		virtual bool TryLoadImpl(Mesh* mesh, bool isConvex, const Vector3& scale, List<uint8_t>& data) = 0;
		virtual void SaveImpl(Mesh* mesh, bool isConvex, const Vector3& scale, List<uint8_t>& data) = 0;

	private:
		static PhysicsShapeCache* s_Instance;
	};
}

