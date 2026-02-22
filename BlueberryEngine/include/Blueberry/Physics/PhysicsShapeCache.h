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
		static void* GetShape(Mesh* mesh, const bool& isConvex, const Vector3& scale);
		/*static void* GetMeshShape(Mesh* mesh, const Vector3& scale);
		static void* GetConvexShape(Mesh* mesh);
		static void Bake(Mesh* mesh, std::ofstream& stream);
		static void Bake(Mesh* mesh, const Vector3& scale);
		static void Load(Mesh* mesh, std::ifstream& stream);*/

	protected:
		virtual void ClearImpl(Mesh* mesh) = 0;
		virtual bool TryLoadImpl(Mesh* mesh, const bool& isConvex, const Vector3& scale, List<uint8_t>& data) = 0;
		virtual void SaveImpl(Mesh* mesh, const bool& isConvex, const Vector3& scale, List<uint8_t>& data) = 0;

	private:
		static inline PhysicsShapeCache* s_Instance = nullptr;
	};
}

