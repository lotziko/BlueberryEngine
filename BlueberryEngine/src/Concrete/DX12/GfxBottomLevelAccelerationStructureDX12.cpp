#include "GfxBottomLevelAccelerationStructureDX12.h"

#include "..\Windows\WindowsHelper.h"
#include "GfxDeviceDX12.h"
#include "GfxBufferDX12.h"

namespace Blueberry
{
	GfxBottomLevelAccelerationStructureDX12::GfxBottomLevelAccelerationStructureDX12(GfxDeviceDX12* device) : m_GfxDevice(device)
	{
	}

	GfxBottomLevelAccelerationStructureDX12::~GfxBottomLevelAccelerationStructureDX12()
	{
		m_GfxDevice->Release(m_Resource);
	}

	bool GfxBottomLevelAccelerationStructureDX12::Initialize(const BottomLevelAccelerationStructureProperties& properties)
	{
		GfxBufferDX12* dxVertexBuffer = static_cast<GfxBufferDX12*>(properties.vertexBuffer);
		GfxBufferDX12* dxIndexBuffer = static_cast<GfxBufferDX12*>(properties.indexBuffer);

		List<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs(properties.subMeshCount);
		for (uint32_t i = 0; i < properties.subMeshCount; ++i)
		{
			const BottomLevelAccelerationStructureSubMesh& submesh = properties.subMeshes[i];
			D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
			geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
			geometryDesc.Triangles.IndexBuffer = dxIndexBuffer->GetResource()->GetGPUVirtualAddress() + submesh.indexStart * sizeof(uint32_t);
			geometryDesc.Triangles.IndexCount = submesh.indexCount;
			geometryDesc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
			geometryDesc.Triangles.Transform3x4 = 0;
			geometryDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
			geometryDesc.Triangles.VertexCount = dxVertexBuffer->GetElementCount();
			geometryDesc.Triangles.VertexBuffer.StartAddress = dxVertexBuffer->GetResource()->GetGPUVirtualAddress();
			geometryDesc.Triangles.VertexBuffer.StrideInBytes = dxVertexBuffer->GetElementSize();
			geometryDesc.Flags = submesh.isOpaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
			geometryDescs[i] = geometryDesc;
		}

		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bottomLevelInputs = {};
		bottomLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		bottomLevelInputs.Flags = buildFlags;
		bottomLevelInputs.NumDescs = properties.subMeshCount;
		bottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
		bottomLevelInputs.pGeometryDescs = geometryDescs.data();
		m_GfxDevice->GetDxrDevice()->GetRaytracingAccelerationStructurePrebuildInfo(&bottomLevelInputs, &bottomLevelPrebuildInfo);

		if (bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes == 0 || bottomLevelPrebuildInfo.ScratchDataSizeInBytes == 0)
		{
			BB_ERROR("Failed to create acceleration structure.");
			return false;
		}

		D3D12_RESOURCE_DESC resourceDesc = {};
		resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		resourceDesc.Width = bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes;
		resourceDesc.Height = 1;
		resourceDesc.DepthOrArraySize = 1;
		resourceDesc.MipLevels = 1;
		resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
		resourceDesc.SampleDesc.Count = 1;
		resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

		HRESULT hr = m_GfxDevice->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&m_Resource));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create acceleration structure."));
			return false;
		}

		D3D12_RESOURCE_DESC scratchResourceDesc = {};
		scratchResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		scratchResourceDesc.Width = bottomLevelPrebuildInfo.ScratchDataSizeInBytes;
		scratchResourceDesc.Height = 1;
		scratchResourceDesc.DepthOrArraySize = 1;
		scratchResourceDesc.MipLevels = 1;
		scratchResourceDesc.Format = DXGI_FORMAT_UNKNOWN;
		scratchResourceDesc.SampleDesc.Count = 1;
		scratchResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		scratchResourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		
		hr = m_GfxDevice->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &scratchResourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_ScratchResource));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create scratch acceleration structure."));
			return false;
		}

		m_GfxDevice->GetCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_ScratchResource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
		dxVertexBuffer->SetState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		dxIndexBuffer->SetState(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc = {};
		bottomLevelBuildDesc.Inputs = bottomLevelInputs;
		bottomLevelBuildDesc.ScratchAccelerationStructureData = m_ScratchResource->GetGPUVirtualAddress();
		bottomLevelBuildDesc.DestAccelerationStructureData = m_Resource->GetGPUVirtualAddress();

		m_GfxDevice->GetDxrCommandList()->BuildRaytracingAccelerationStructure(&bottomLevelBuildDesc, 0, nullptr);
		m_GfxDevice->Release(m_ScratchResource);	// TODO scratch buffer
		m_ScratchResource = nullptr;

		m_GeometryData.vertexBufferAddress = dxVertexBuffer->GetResource()->GetGPUVirtualAddress();
		m_GeometryData.indexBufferAddress = dxIndexBuffer->GetResource()->GetGPUVirtualAddress();
		m_GeometryData.vertexStride = properties.vertexStride;
		m_GeometryData.normalOffset = properties.normalOffset;
		m_GeometryData.uv0Offset = properties.uv0Offset;

		return true;
	}
}