#include "Blueberry\Physics\PhysicsShapeCache.h"

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\Mesh.h"

#include <Jolt\Jolt.h>
#include <Jolt\Core\StreamOut.h>
#include <Jolt\Physics\Collision\Shape\MeshShape.h>
#include <Jolt\Physics\Collision\Shape\ConvexHullShape.h>
#include <fstream>

namespace Blueberry
{
	PhysicsShapeCache* PhysicsShapeCache::s_Instance = nullptr;
	static Dictionary<size_t, JPH::Ref<JPH::Shape>> s_Shapes = {};

	class JoltStreamOut : public JPH::StreamOut
	{
	public:
		JoltStreamOut(List<uint8_t>& data) : m_Data(data)
		{
		}

		virtual void WriteBytes(const void* inData, size_t inNumBytes) override
		{
			size_t oldSize = m_Data.size();
			m_Data.resize(oldSize + inNumBytes);
			memcpy(m_Data.data() + oldSize, inData, inNumBytes);
		}

		virtual bool IsFailed() const override
		{
			return false;
		}

	private:
		List<uint8_t>& m_Data;
	};

	class JoltStreamIn : public JPH::StreamIn
	{
	public:
		JoltStreamIn(List<uint8_t>& data) : m_Data(data)
		{
		}

		virtual void ReadBytes(void* outData, size_t inNumBytes) override
		{
			memcpy(outData, m_Data.data() + m_Position, inNumBytes);
			m_Position += inNumBytes;
		}

		virtual bool IsEOF() const override
		{
			return m_Position > m_Data.size();
		}

		virtual bool IsFailed() const override
		{
			return false;
		}

	private:
		List<uint8_t>& m_Data;
		size_t m_Position = 0;
	};

	size_t GetKey(Mesh* mesh, bool isConvex, const Vector3& scale)
	{
		size_t mask = (scale.x > 0 ? 1 : 0) | (scale.y > 0 ? 2 : 0) | (scale.z > 0 ? 4 : 0);
		return static_cast<size_t>(mesh->GetObjectId()) | (isConvex ? 1ull << 32 : 0) | mask << 33;
	}

	void PhysicsShapeCache::Initialize(PhysicsShapeCache* shapeCache)
	{
		s_Instance = shapeCache;
	}

	void PhysicsShapeCache::Clear(Mesh* mesh)
	{
		s_Instance->ClearImpl(mesh);
		size_t key = mesh->GetObjectId();
		List<size_t> keysToRemove;
		for (auto& pair : s_Shapes)
		{
			if ((pair.first & key) == key)
			{
				keysToRemove.push_back(pair.first);
			}
		}
		for (size_t key : keysToRemove)
		{
			s_Shapes.erase(key);
		}
	}

	void* PhysicsShapeCache::GetShape(Mesh* mesh, bool isConvex, const Vector3& scale)
	{
		size_t key = GetKey(mesh, isConvex, scale);
		auto it = s_Shapes.find(key);
		if (it != s_Shapes.end())
		{
			return it->second.GetPtr();
		}

		List<uint8_t> data;
		if (s_Instance->TryLoadImpl(mesh, isConvex, scale, data))
		{
			JoltStreamIn joltStream(data);
			JPH::Shape::ShapeResult result = JPH::Shape::sRestoreFromBinaryState(joltStream);
			if (result.IsValid())
			{
				s_Shapes.insert_or_assign(key, result.Get());
				return result.Get().GetPtr();
			}
			else
			{
				BB_ERROR("Failed to load a shape.");
			}
		}
		else
		{
			if (isConvex)
			{
				JPH::Array<JPH::Vec3> points = {};
				Vector3* vertices = mesh->GetVertices();
				uint32_t* indices = mesh->GetIndices();
				points.resize(mesh->GetIndexCount());

				Vector3 bakeScale = Vector3(scale.x > 0.0f ? 1.0f : -1.0f, scale.y > 0.0f ? 1.0f : -1.0f, scale.z > 0.0f ? 1.0f : -1.0f);
				for (uint32_t i = 0; i < mesh->GetIndexCount(); ++i)
				{
					Vector3 vertex = vertices[indices[i]] * bakeScale;
					points[i] = JPH::Vec3(vertex.x, vertex.y, vertex.z);
				}

				JoltStreamOut joltStream(data);
				JPH::ConvexHullShapeSettings settings(points);
				JPH::Shape::ShapeResult result = settings.Create();
				if (result.IsValid())
				{
					result.Get()->SaveBinaryState(joltStream);
					s_Instance->SaveImpl(mesh, isConvex, scale, data);
					s_Shapes.insert_or_assign(key, result.Get());
					return result.Get().GetPtr();
				}
			}
			else
			{
				JPH::TriangleList triangles = {};
				Vector3* vertices = mesh->GetVertices();
				uint32_t* indices = mesh->GetIndices();
				triangles.resize(mesh->GetIndexCount() / 3);
				JPH::Triangle* trianglePtr = triangles.data();

				Vector3 bakeScale = Vector3(scale.x > 0.0f ? 1.0f : -1.0f, scale.y > 0.0f ? 1.0f : -1.0f, scale.z > 0.0f ? 1.0f : -1.0f);
				if (scale.x * scale.y * scale.z > 0.0f)
				{
					for (uint32_t i = 0; i < mesh->GetIndexCount(); i += 3)
					{
						Vector3 vertex1 = vertices[indices[i]] * bakeScale;
						Vector3 vertex2 = vertices[indices[i + 1]] * bakeScale;
						Vector3 vertex3 = vertices[indices[i + 2]] * bakeScale;
						*trianglePtr = JPH::Triangle(JPH::Vec3(vertex1.x, vertex1.y, vertex1.z), JPH::Vec3(vertex2.x, vertex2.y, vertex2.z), JPH::Vec3(vertex3.x, vertex3.y, vertex3.z));
						++trianglePtr;
					}
				}
				else
				{
					for (uint32_t i = 0; i < mesh->GetIndexCount(); i += 3)
					{
						Vector3 vertex1 = vertices[indices[i]] * bakeScale;
						Vector3 vertex2 = vertices[indices[i + 1]] * bakeScale;
						Vector3 vertex3 = vertices[indices[i + 2]] * bakeScale;
						*trianglePtr = JPH::Triangle(JPH::Vec3(vertex1.x, vertex1.y, vertex1.z), JPH::Vec3(vertex3.x, vertex3.y, vertex3.z), JPH::Vec3(vertex2.x, vertex2.y, vertex2.z));
						++trianglePtr;
					}
				}

				JoltStreamOut joltStream(data);
				JPH::MeshShapeSettings settings(triangles);
				JPH::Shape::ShapeResult result = settings.Create();
				if (result.IsValid())
				{
					result.Get()->SaveBinaryState(joltStream);
					s_Instance->SaveImpl(mesh, isConvex, scale, data);
					s_Shapes.insert_or_assign(key, result.Get());
					return result.Get().GetPtr();
				}
			}
		}
		return nullptr;
	}
}
