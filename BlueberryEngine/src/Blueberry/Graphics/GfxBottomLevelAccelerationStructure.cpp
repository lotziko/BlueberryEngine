#include "Blueberry\Graphics\GfxBottomLevelAccelerationStructure.h"

#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	Dictionary<ObjectId, std::pair<uint32_t, GfxBottomLevelAccelerationStructure*>> GfxBottomLevelAccelerationStructure::s_AccelerationStructures = {};

	GfxBottomLevelAccelerationStructure* GfxBottomLevelAccelerationStructure::Get(Mesh* mesh, bool opaqueMask[16])
	{
		ObjectId key = mesh->GetObjectId();
		uint32_t updateCount = mesh->GetUpdateCount();

		auto it = s_AccelerationStructures.find(key);
		if (it != s_AccelerationStructures.end())
		{
			if (it->second.first != updateCount)
			{
				delete it->second.second;
				GfxBottomLevelAccelerationStructure* accelerationStructure = CreateAccelerationStructure(mesh, opaqueMask);
				s_AccelerationStructures.insert_or_assign(key, std::make_pair(updateCount, accelerationStructure));
				return accelerationStructure;
			}
			return it->second.second;
		}
		else
		{
			GfxBottomLevelAccelerationStructure* accelerationStructure = CreateAccelerationStructure(mesh, opaqueMask);
			s_AccelerationStructures.insert_or_assign(key, std::make_pair(updateCount, accelerationStructure));
			return accelerationStructure;
		}
	}

	GfxBottomLevelAccelerationStructure* GfxBottomLevelAccelerationStructure::CreateAccelerationStructure(Mesh* mesh, bool opaqueMask[16])
	{
		GfxBottomLevelAccelerationStructure* accelerationStructure;
		const VertexLayout& layout = mesh->GetLayout();
		BottomLevelAccelerationStructureProperties accelerationStructureProperties = {};
		accelerationStructureProperties.vertexBuffer = mesh->GetVertexBuffer();
		accelerationStructureProperties.indexBuffer = mesh->GetIndexBuffer();
		accelerationStructureProperties.vertexStride = layout.GetSize();
		accelerationStructureProperties.normalOffset = layout.GetOffset(VertexAttribute::Normal);
		accelerationStructureProperties.tangentOffset = layout.GetOffset(VertexAttribute::Tangent);
		accelerationStructureProperties.uv0Offset = layout.GetOffset(VertexAttribute::Texcoord0);
		for (uint32_t i = 0, n = std::min(mesh->GetSubMeshCount(), 16u); i < n; ++i)
		{
			const SubMeshData& subMeshData = mesh->GetSubMesh(i);
			BottomLevelAccelerationStructureSubMesh subMesh = {};
			subMesh.indexStart = subMeshData.GetIndexStart();
			subMesh.indexCount = subMeshData.GetIndexCount();
			subMesh.isOpaque = opaqueMask[i];
			accelerationStructureProperties.subMeshes[i] = subMesh;
		}
		accelerationStructureProperties.subMeshCount = mesh->GetSubMeshCount();
		GfxDevice::CreateBottomLevelAccelerationStructure(accelerationStructureProperties, accelerationStructure);
		return accelerationStructure;
	}
}