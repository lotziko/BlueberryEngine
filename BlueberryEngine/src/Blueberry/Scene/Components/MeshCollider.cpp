#include "Blueberry\Scene\Components\MeshCollider.h"

#include "Blueberry\Scene\Entity.h"
#include "Blueberry\Scene\Components\Transform.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Physics\PhysicsShapeCache.h"
#include "Blueberry\Core\ClassDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(MeshCollider, Collider)
	{
		DEFINE_BASE_FIELDS(MeshCollider, Collider)
		DEFINE_FIELD(MeshCollider, m_Mesh, BindingType::ObjectPtr, FieldOptions().SetObjectType(Mesh::Type))
	}

	JPH::Shape* MeshCollider::GetShape()
	{
		return static_cast<JPH::Shape*>(PhysicsShapeCache::GetShape(m_Mesh.Get()));
	}
}