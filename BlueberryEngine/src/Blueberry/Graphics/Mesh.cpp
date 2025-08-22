#include "Blueberry\Graphics\Mesh.h"

#include "Blueberry\Core\ClassDB.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Tools\CRCHelper.h"

#include <mikktspace\mikktspace.h>

namespace Blueberry
{
	DATA_DEFINITION(SubMeshData)
	{
		DEFINE_FIELD(SubMeshData, m_IndexStart, BindingType::Int, {})
		DEFINE_FIELD(SubMeshData, m_IndexCount, BindingType::Int, {})
	}

	OBJECT_DEFINITION(Mesh, Object)
	{
		DEFINE_BASE_FIELDS(Mesh, Object)
		DEFINE_FIELD(Mesh, m_VertexData, BindingType::FloatList, {})
		DEFINE_FIELD(Mesh, m_IndexData, BindingType::IntList, {})
		DEFINE_FIELD(Mesh, m_VertexCount, BindingType::Int, {})
		DEFINE_FIELD(Mesh, m_IndexCount, BindingType::Int, {})
		DEFINE_FIELD(Mesh, m_SubMeshes, BindingType::DataList, FieldOptions().SetObjectType(SubMeshData::Type))
		DEFINE_FIELD(Mesh, m_Layout, BindingType::Raw, FieldOptions().SetSize(sizeof(VertexLayout)))
		DEFINE_FIELD(Mesh, m_Bounds, BindingType::AABB, {})
	}

	const uint32_t& SubMeshData::GetIndexStart() const
	{
		return m_IndexStart;
	}

	void SubMeshData::SetIndexStart(const uint32_t& indexStart)
	{
		m_IndexStart = indexStart;
	}

	const uint32_t& SubMeshData::GetIndexCount() const
	{
		return m_IndexCount;
	}

	void SubMeshData::SetIndexCount(const uint32_t& indexCount)
	{
		m_IndexCount = indexCount;
	}

	const int VERTICES_BIT = 1;
	const int NORMALS_BIT = 2;
	const int TANGENTS_BIT = 4;
	const int UV0_BIT = 8;

	Mesh::~Mesh()
	{
		if (m_VertexBuffer != nullptr)
		{
			delete m_VertexBuffer;
		}
		if (m_IndexBuffer != nullptr)
		{
			delete m_IndexBuffer;
		}
	}

	const uint32_t& Mesh::GetVertexCount()
	{
		return m_VertexCount;
	}

	const uint32_t& Mesh::GetIndexCount()
	{
		return m_IndexCount;
	}

	const uint32_t Mesh::GetSubMeshCount()
	{
		return static_cast<uint32_t>(m_SubMeshes.size());
	}

	const SubMeshData& Mesh::GetSubMesh(const uint32_t& index)
	{
		return m_SubMeshes[index];
	}

	Vector3* Mesh::GetVertices()
	{
		if (m_Vertices.size() == 0 && m_VertexData.size() > 0)
		{
			m_Vertices.resize(m_VertexCount);
			uint32_t vertexSize = m_Layout.GetSize() / sizeof(float);
			float* bufferPtr = m_VertexData.data();
			Vector3* verticesPtr = m_Vertices.data();
			for (uint32_t i = 0; i < m_VertexCount; ++i)
			{
				memcpy(verticesPtr, bufferPtr, sizeof(Vector3));
				bufferPtr += vertexSize;
				verticesPtr += 1;
			}
		}
		return m_Vertices.data();
	}

	Vector3* Mesh::GetNormals()
	{
		if (!m_Layout.Has(VertexAttribute::Normal))
		{
			return nullptr;
		}
		if (m_Normals.size() == 0 && m_VertexData.size() > 0)
		{
			m_Normals.resize(m_VertexCount);
			uint32_t vertexSize = m_Layout.GetSize() / sizeof(float);
			float* bufferPtr = m_VertexData.data() + m_Layout.GetOffset(VertexAttribute::Normal) / sizeof(float);
			Vector3* normalsPtr = m_Normals.data();
			for (uint32_t i = 0; i < m_VertexCount; ++i)
			{
				memcpy(normalsPtr, bufferPtr, sizeof(Vector3));
				bufferPtr += vertexSize;
				normalsPtr += 1;
			}
		}
		return m_Normals.data();
	}

	Vector4* Mesh::GetTangents()
	{
		if (!m_Layout.Has(VertexAttribute::Tangent))
		{
			return nullptr;
		}
		if (m_Tangents.size() == 0 && m_VertexData.size() > 0)
		{
			m_Tangents.resize(m_VertexCount);
			uint32_t vertexSize = m_Layout.GetSize() / sizeof(float);
			float* bufferPtr = m_VertexData.data() + m_Layout.GetOffset(VertexAttribute::Tangent) / sizeof(float);
			Vector4* tangentsPtr = m_Tangents.data();
			for (uint32_t i = 0; i < m_VertexCount; ++i)
			{
				memcpy(tangentsPtr, bufferPtr, sizeof(Vector4));
				bufferPtr += vertexSize;
				tangentsPtr += 1;
			}
		}
		return m_Tangents.data();
	}

	Color* Mesh::GetColors()
	{
		if (!m_Layout.Has(VertexAttribute::Color))
		{
			return nullptr;
		}
		if (m_Colors.size() == 0 && m_VertexData.size() > 0)
		{
			m_Colors.resize(m_VertexCount);
			uint32_t vertexSize = m_Layout.GetSize() / sizeof(float);
			float* bufferPtr = m_VertexData.data() + m_Layout.GetOffset(VertexAttribute::Color) / sizeof(float);
			Color* colorsPtr = m_Colors.data();
			for (uint32_t i = 0; i < m_VertexCount; ++i)
			{
				memcpy(colorsPtr, bufferPtr, sizeof(Color));
				bufferPtr += vertexSize;
				colorsPtr += 1;
			}
		}
		return m_Colors.data();
	}

	uint32_t* Mesh::GetIndices()
	{
		return m_IndexData.data();
	}

	float* Mesh::GetUVs(const int& channel)
	{
		VertexAttribute attribute = static_cast<VertexAttribute>(4 + channel);
		if (!m_Layout.Has(attribute))
		{
			return nullptr;
		}
		List<float>& uvs = m_UVs[channel];
		if (uvs.size() == 0 && m_VertexData.size() > 0)
		{
			uint32_t uvByteSize = m_Layout.GetSize(attribute);
			uint32_t uvSize = uvByteSize / sizeof(float);
			uint32_t vertexSize = m_Layout.GetSize() / sizeof(float);
			uvs.resize(m_VertexCount * uvSize);
			float* bufferPtr = m_VertexData.data() + m_Layout.GetOffset(attribute) / sizeof(float);
			float* uvsPtr = uvs.data();
			for (uint32_t i = 0; i < m_VertexCount; ++i)
			{
				memcpy(uvsPtr, bufferPtr, uvByteSize);
				bufferPtr += vertexSize;
				uvsPtr += uvSize;
			}
		}
		return uvs.data();
	}

	uint32_t Mesh::GetUVSize(const int& channel)
	{
		if (channel < 0 || channel >= 4)
		{
			return 0;
		}
		return m_UVs[channel].size() / m_VertexCount;
	}

	void Mesh::SetVertices(const Vector3* vertices, const uint32_t& vertexCount)
	{
		m_VertexCount = vertexCount;
		m_Vertices.resize(vertexCount);
		memcpy(m_Vertices.data(), vertices, sizeof(Vector3) * vertexCount);
		m_Layout.Append(VertexAttribute::Position, sizeof(Vector3));
		m_BufferIsDirty = true;
	}

	void Mesh::SetNormals(const Vector3* normals, const uint32_t& vertexCount)
	{
		m_Normals.resize(vertexCount);
		memcpy(m_Normals.data(), normals, sizeof(Vector3) * vertexCount);
		m_Layout.Append(VertexAttribute::Normal, sizeof(Vector3));
		m_BufferIsDirty = true;
	}

	void Mesh::SetTangents(const Vector4* tangents, const uint32_t& vertexCount)
	{
		m_Tangents.resize(vertexCount);
		memcpy(m_Tangents.data(), tangents, sizeof(Vector4) * vertexCount);
		m_Layout.Append(VertexAttribute::Tangent, sizeof(Vector4));
		m_BufferIsDirty = true;
	}

	void Mesh::SetColors(const Color* colors, const uint32_t& vertexCount)
	{
		m_Colors.resize(vertexCount);
		memcpy(m_Colors.data(), colors, sizeof(Color) * vertexCount);
		m_Layout.Append(VertexAttribute::Color, sizeof(Color));
		m_BufferIsDirty = true;
	}

	void Mesh::SetIndices(const uint32_t* indices, const uint32_t& indexCount)
	{
		m_IndexCount = indexCount;
		m_IndexData.resize(indexCount);
		memcpy(m_IndexData.data(), indices, sizeof(uint32_t) * indexCount);
		m_BufferIsDirty = true;
	}

	void Mesh::SetUVs(const int& channel, const Vector2* uvs, const uint32_t& uvCount)
	{
		if (channel < 0 || channel >= 4)
		{
			return;
		}
		m_UVs[channel].resize(uvCount * 2);
		memcpy(m_UVs[channel].data(), uvs, sizeof(Vector2) * uvCount);
		m_Layout.Append(static_cast<VertexAttribute>(4 + channel), sizeof(Vector2));
		m_BufferIsDirty = true;
	}

	void Mesh::SetUVs(const int& channel, const Vector3* uvs, const uint32_t& uvCount)
	{
		if (channel < 0 || channel >= 4)
		{
			return;
		}
		m_UVs[channel].resize(uvCount * 3);
		memcpy(m_UVs[channel].data(), uvs, sizeof(Vector3) * uvCount);
		m_Layout.Append(static_cast<VertexAttribute>(4 + channel), sizeof(Vector3));
		m_BufferIsDirty = true;
	}

	void Mesh::SetUVs(const int& channel, const Vector4* uvs, const uint32_t& uvCount)
	{
		if (channel < 0 || channel >= 4)
		{
			return;
		}
		m_UVs[channel].resize(uvCount * 4);
		memcpy(m_UVs[channel].data(), uvs, sizeof(Vector4) * uvCount);
		m_Layout.Append(static_cast<VertexAttribute>(4 + channel), sizeof(Vector4));
		m_BufferIsDirty = true;
	}

	void Mesh::SetSubMesh(const uint32_t& index, const SubMeshData& data)
	{
		if (index >= m_SubMeshes.size())
		{
			m_SubMeshes.resize(index + 1);
		}
		m_SubMeshes[index] = data;
	}

	// Based on https://www.turais.de/using-mikktspace-in-your-project/
	class TangentGenerator
	{
	public:
		void Generate(Mesh* mesh)
		{
			m_Iface.m_getNumFaces = GetNumFaces;
			m_Iface.m_getNumVerticesOfFace = GetNumVerticesOfFace;

			m_Iface.m_getNormal = GetNormal;
			m_Iface.m_getPosition = GetPosition;
			m_Iface.m_getTexCoord = GetTexCoords;
			m_Iface.m_setTSpaceBasic = SetTspaceBasic;

			m_Context.m_pInterface = &m_Iface;
			m_Context.m_pUserData = mesh;

			genTangSpaceDefault(&this->m_Context);
		}

	private:
		static int GetNumFaces(const SMikkTSpaceContext* context)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			return mesh->m_IndexCount / 3;
		}

		static int GetNumVerticesOfFace(const SMikkTSpaceContext* context, int iFace)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			return 3;
		}

		static void GetPosition(const SMikkTSpaceContext* context, float outpos[], int iFace, int iVert)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			Vector3 position = mesh->m_Vertices[mesh->m_IndexData[iFace * 3 + iVert]];

			outpos[0] = position.x;
			outpos[1] = position.y;
			outpos[2] = position.z;
		}

		static void GetNormal(const SMikkTSpaceContext* context, float outnormal[], int iFace, int iVert)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			Vector3 normal = mesh->m_Normals[mesh->m_IndexData[iFace * 3 + iVert]];

			outnormal[0] = normal.x;
			outnormal[1] = normal.y;
			outnormal[2] = normal.z;
		}

		static void GetTexCoords(const SMikkTSpaceContext* context, float outuv[], int iFace, int iVert)
		{
			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			uint32_t stride = mesh->m_Layout.GetSize(VertexAttribute::Texcoord0) / sizeof(float);
			uint32_t offset = mesh->m_IndexData[iFace * 3 + iVert] * stride;
			
			outuv[0] = mesh->m_UVs[0][offset];
			outuv[1] = mesh->m_UVs[0][offset + 1];
		}

		static void SetTspaceBasic(const SMikkTSpaceContext *context, const float tangentu[], float fSign, int iFace, int iVert)
		{
			Vector4 tangent;
			tangent.x = tangentu[0];
			tangent.y = tangentu[1];
			tangent.z = tangentu[2];
			tangent.w = fSign;

			Mesh* mesh = static_cast<Mesh*> (context->m_pUserData);
			mesh->m_Tangents[mesh->m_IndexData[iFace * 3 + iVert]] = tangent;
		}

	private:
		SMikkTSpaceInterface m_Iface = {};
		SMikkTSpaceContext m_Context = {};
	};
	
	void Mesh::GenerateTangents()
	{
		m_Tangents.resize(m_VertexCount);
		TangentGenerator generator;
		generator.Generate(this);
		m_Layout.Append(VertexAttribute::Tangent, sizeof(Vector4));
		m_BufferIsDirty = true;
	}

	const Topology& Mesh::GetTopology()
	{
		return m_Topology;
	}

	void Mesh::SetTopology(const Topology& topology)
	{
		m_Topology = topology;
	}

	const AABB& Mesh::GetBounds()
	{
		return m_Bounds;
	}

	void Mesh::Apply()
	{
		if (m_BufferIsDirty)
		{
			m_Layout.Apply();
			uint32_t vertexSize = m_Layout.GetSize() / sizeof(float);
			uint32_t vertexBufferSize = vertexSize * m_VertexCount;
			uint32_t vertexOffset = 0;

			if (vertexBufferSize != m_VertexData.size())
			{
				m_VertexData.resize(vertexBufferSize);
			}

			float* bufferPtr = m_VertexData.data();

			if (m_Layout.Has(VertexAttribute::Position))
			{
				AABB::CreateFromPoints(m_Bounds, m_VertexCount, m_Vertices.data(), sizeof(Vector3));
				Vector3* verticesPtr = m_Vertices.data();
				for (uint32_t i = 0; i < m_VertexCount; ++i)
				{
					memcpy(bufferPtr, verticesPtr, sizeof(Vector3));
					bufferPtr += vertexSize;
					verticesPtr += 1;
				}
				vertexOffset += sizeof(Vector3) / sizeof(float);
			}

			if (m_Layout.Has(VertexAttribute::Normal))
			{
				bufferPtr = m_VertexData.data() + vertexOffset;
				Vector3* normalsPtr = m_Normals.data();
				for (uint32_t i = 0; i < m_VertexCount; ++i)
				{
					memcpy(bufferPtr, normalsPtr, sizeof(Vector3));
					bufferPtr += vertexSize;
					normalsPtr += 1;
				}
				vertexOffset += sizeof(Vector3) / sizeof(float);
			}

			if (m_Layout.Has(VertexAttribute::Tangent))
			{
				bufferPtr = m_VertexData.data() + vertexOffset;
				Vector4* tangentsPtr = m_Tangents.data();
				for (uint32_t i = 0; i < m_VertexCount; ++i)
				{
					memcpy(bufferPtr, tangentsPtr, sizeof(Vector4));
					bufferPtr += vertexSize;
					tangentsPtr += 1;
				}
				vertexOffset += sizeof(Vector4) / sizeof(float);
			}

			if (m_Layout.Has(VertexAttribute::Color))
			{
				bufferPtr = m_VertexData.data() + vertexOffset;
				Color* colorsPtr = m_Colors.data();
				for (uint32_t i = 0; i < m_VertexCount; ++i)
				{
					memcpy(bufferPtr, colorsPtr, sizeof(Color));
					bufferPtr += vertexSize;
					colorsPtr += 1;
				}
				vertexOffset += sizeof(Color) / sizeof(float);
			}

			for (uint32_t channel = 0; channel < 4; ++channel)
			{
				VertexAttribute attribute = static_cast<VertexAttribute>(4 + channel);
				if (m_Layout.Has(attribute))
				{
					bufferPtr = m_VertexData.data() + vertexOffset;
					uint32_t uvSize = m_Layout.GetSize(attribute);
					uint32_t uvChannelCount = uvSize / sizeof(float);
					float* uvsPtr = m_UVs[channel].data();
					for (uint32_t i = 0; i < m_VertexCount; ++i)
					{
						memcpy(bufferPtr, uvsPtr, uvSize);
						bufferPtr += vertexSize;
						uvsPtr += uvChannelCount;
					}
					vertexOffset += uvChannelCount;
				}
			}
		}

		if (m_VertexBuffer != nullptr)
		{
			delete m_VertexBuffer;
		}
		if (m_IndexBuffer != nullptr)
		{
			delete m_IndexBuffer;
		}

		BufferProperties vertexBufferProperties = {};
		vertexBufferProperties.type = BufferType::Vertex;
		vertexBufferProperties.elementCount = m_VertexCount;
		vertexBufferProperties.elementSize = m_Layout.GetSize();
		vertexBufferProperties.data = m_VertexData.data();
		vertexBufferProperties.dataSize = m_VertexCount * vertexBufferProperties.elementSize;
		vertexBufferProperties.isWritable = false;
		GfxDevice::CreateBuffer(vertexBufferProperties, m_VertexBuffer);

		BufferProperties indexBufferProperties = {};
		indexBufferProperties.type = BufferType::Index;
		indexBufferProperties.elementCount = m_IndexCount;
		indexBufferProperties.elementSize = sizeof(uint32_t);
		indexBufferProperties.data = m_IndexData.data();
		indexBufferProperties.dataSize = m_IndexCount * indexBufferProperties.elementSize;
		indexBufferProperties.isWritable = false;
		GfxDevice::CreateBuffer(indexBufferProperties, m_IndexBuffer);
	}

	const VertexLayout& Mesh::GetLayout()
	{
		return m_Layout;
	}

	Mesh* Mesh::Create()
	{
		Mesh* mesh = Object::Create<Mesh>();
		return mesh;
	}
}