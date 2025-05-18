#include "PhysicsShapeCache.h"

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\Mesh.h"

#include <Jolt\Jolt.h>
#include <Jolt\Core\StreamOut.h>
#include <Jolt\Physics\Collision\Shape\MeshShape.h>
#include <fstream>

namespace Blueberry
{
	static Dictionary<ObjectId, JPH::Shape::ShapeResult> s_ShapeResults = {};

	class JoltStreamOut : public JPH::StreamOut
	{
	public:
		JoltStreamOut(std::ofstream& stream) : m_Stream(stream)
		{
		}

		virtual void WriteBytes(const void* inData, size_t inNumBytes) override
		{
			m_Stream.write(static_cast<const char*>(inData), inNumBytes);
		}

		virtual bool IsFailed() const override
		{
			return false;
		}

	private:
		std::ofstream& m_Stream;
	};

	class JoltStreamIn : public JPH::StreamIn
	{
	public:
		JoltStreamIn(std::ifstream& stream) : m_Stream(stream)
		{
		}

		virtual void ReadBytes(void* outData, size_t inNumBytes) override
		{
			m_Stream.read(static_cast<char*>(outData), inNumBytes);
		}

		virtual bool IsEOF() const override
		{
			return m_Stream.eof();
		}

		virtual bool IsFailed() const override
		{
			return false;
		}
	private:
		std::ifstream& m_Stream;
	};

	void* PhysicsShapeCache::GetShape(Mesh* mesh)
	{
		auto it = s_ShapeResults.find(mesh->GetObjectId());
		if (it != s_ShapeResults.end())
		{
			return it->second.Get().GetPtr();
		}
		return nullptr;
	}

	void PhysicsShapeCache::Bake(Mesh* mesh, std::ofstream& stream)
	{
		JPH::TriangleList triangles;
		auto& vertices = mesh->GetVertices();
		auto& indices = mesh->GetIndices();
		triangles.resize(indices.size() / 3);
		JPH::Triangle* trianglePtr = triangles.data();

		for (uint32_t i = 0; i < indices.size(); i += 3)
		{
			Vector3 vertex1 = vertices[i];
			Vector3 vertex2 = vertices[i + 1];
			Vector3 vertex3 = vertices[i + 2];
			*trianglePtr = JPH::Triangle(JPH::Vec3(vertex1.x, vertex1.y, vertex1.z), JPH::Vec3(vertex2.x, vertex2.y, vertex2.z), JPH::Vec3(vertex3.x, vertex3.y, vertex3.z));
			++trianglePtr;
		}

		JoltStreamOut joltStream(stream);
		JPH::MeshShapeSettings settings(triangles);
		JPH::Shape::ShapeResult result = settings.Create();
		s_ShapeResults.insert_or_assign(mesh->GetObjectId(), result);
		result.Get()->SaveBinaryState(joltStream);
	}

	void PhysicsShapeCache::Load(Mesh* mesh, std::ifstream& stream)
	{
		JoltStreamIn joltStream(stream);
		s_ShapeResults.insert_or_assign(mesh->GetObjectId(), JPH::Shape::sRestoreFromBinaryState(joltStream));
	}
}
