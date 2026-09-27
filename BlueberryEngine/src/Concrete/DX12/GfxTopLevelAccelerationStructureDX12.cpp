#include "GfxTopLevelAccelerationStructureDX12.h"

#include "Blueberry\Graphics\Material.h"

#include "..\Windows\WindowsHelper.h"
#include "GfxDeviceDX12.h"
#include "GfxBufferDX12.h"
#include "GfxBottomLevelAccelerationStructureDX12.h"

namespace Blueberry
{
	GfxTopLevelAccelerationStructureDX12::GfxTopLevelAccelerationStructureDX12(GfxDeviceDX12* device) : m_GfxDevice(device), m_Device(device->GetDevice())
	{
	}

	GfxTopLevelAccelerationStructureDX12::~GfxTopLevelAccelerationStructureDX12()
	{
		m_GfxDevice->Release(m_Resource);
		m_GfxDevice->Release(m_ScratchResource);
		m_ShaderResourceView.Free();
	}

	void GfxTopLevelAccelerationStructureDX12::Add(GfxBottomLevelAccelerationStructure* accelerationStructure, const List<ObjectPtr<Material>>& materials, const Matrix& transform)
	{
		auto dxAccelerationStructure = static_cast<GfxBottomLevelAccelerationStructureDX12*>(accelerationStructure);

		UINT materialOffset = static_cast<UINT>(m_InstanceDatas.size());
		for (size_t i = 0; i < materials.size(); ++i)
		{
			GfxRayTracingInstanceDataDX12 instanceData = {};
			instanceData.bottomLevelAccelerationStructure = dxAccelerationStructure;
			instanceData.geometryData = dxAccelerationStructure->m_GeometryData;
			instanceData.material = materials[i].Get();
			m_InstanceDatas.push_back(std::move(instanceData));
		}
		
		D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {};
		instanceDesc.InstanceID = static_cast<UINT>(materials.size());
		instanceDesc.InstanceMask = 1;
		instanceDesc.InstanceContributionToHitGroupIndex = materialOffset;
		instanceDesc.AccelerationStructure = dxAccelerationStructure->m_Resource->GetGPUVirtualAddress();
		instanceDesc.Transform[0][0] = transform._11;
		instanceDesc.Transform[0][1] = transform._21;
		instanceDesc.Transform[0][2] = transform._31;
		instanceDesc.Transform[0][3] = transform._41;
		instanceDesc.Transform[1][0] = transform._12;
		instanceDesc.Transform[1][1] = transform._22;
		instanceDesc.Transform[1][2] = transform._32;
		instanceDesc.Transform[1][3] = transform._42;
		instanceDesc.Transform[2][0] = transform._13;
		instanceDesc.Transform[2][1] = transform._23;
		instanceDesc.Transform[2][2] = transform._33;
		instanceDesc.Transform[2][3] = transform._43;
		m_InstanceDescs.push_back(instanceDesc);
	}

	void GfxTopLevelAccelerationStructureDX12::Clear()
	{
		m_InstanceDatas.clear();
		m_InstanceDescs.clear();
	}

	void GfxTopLevelAccelerationStructureDX12::Build()
	{
		if (m_InstanceDescs.size() == 0)
		{
			return;
		}

		HRESULT hr;
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS topLevelInputs = {};
		topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		topLevelInputs.Flags = buildFlags;
		topLevelInputs.NumDescs = static_cast<UINT>(m_InstanceDescs.size());
		topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;

		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo = {};
		m_GfxDevice->GetDxrDevice()->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &topLevelPrebuildInfo);

		if (topLevelPrebuildInfo.ResultDataMaxSizeInBytes == 0 || topLevelPrebuildInfo.ScratchDataSizeInBytes == 0)
		{
			return;
		}

		if (m_Resource.Get() == nullptr || m_ResourceSize < topLevelPrebuildInfo.ResultDataMaxSizeInBytes)
		{
			if (m_Resource.Get() != nullptr)
			{
				m_GfxDevice->Release(m_Resource);
				m_Resource = nullptr;
			}

			D3D12_RESOURCE_DESC resourceDesc = {};
			resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			resourceDesc.Width = topLevelPrebuildInfo.ResultDataMaxSizeInBytes;
			resourceDesc.Height = 1;
			resourceDesc.DepthOrArraySize = 1;
			resourceDesc.MipLevels = 1;
			resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
			resourceDesc.SampleDesc.Count = 1;
			resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

			hr = m_GfxDevice->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&m_Resource));

			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create acceleration structure."));
				return;
			}

			D3D12_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc = {};
			shaderResourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
			shaderResourceViewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			shaderResourceViewDesc.RaytracingAccelerationStructure.Location = m_Resource->GetGPUVirtualAddress();

			m_ShaderResourceView = m_GfxDevice->GetCbvSrvUavHeap().AllocatePersistent();
			m_Device->CreateShaderResourceView(nullptr, &shaderResourceViewDesc, m_ShaderResourceView.GetCPU());

			m_ResourceSize = resourceDesc.Width;
		}

		if (m_ScratchResource.Get() == nullptr || m_ScratchResourceSize < topLevelPrebuildInfo.ScratchDataSizeInBytes)
		{
			if (m_ScratchResource.Get() != nullptr)
			{
				m_GfxDevice->Release(m_ScratchResource);
				m_ScratchResource = nullptr;
			}

			D3D12_RESOURCE_DESC scratchResourceDesc = {};
			scratchResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			scratchResourceDesc.Width = topLevelPrebuildInfo.ScratchDataSizeInBytes;
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
				return;
			}

			m_GfxDevice->GetCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_ScratchResource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
			m_ScratchResourceSize = scratchResourceDesc.Width;
		}

		D3D12_GPU_VIRTUAL_ADDRESS instanceDescsAdress;
		m_GfxDevice->GetUploadBuffer().UploadBuffer(instanceDescsAdress, m_InstanceDescs.data(), m_InstanceDescs.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), 16ull);

		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc = {};
		topLevelInputs.InstanceDescs = instanceDescsAdress;
		topLevelBuildDesc.Inputs = topLevelInputs;
		topLevelBuildDesc.DestAccelerationStructureData = m_Resource->GetGPUVirtualAddress();
		topLevelBuildDesc.ScratchAccelerationStructureData = m_ScratchResource->GetGPUVirtualAddress();

		m_GfxDevice->GetDxrCommandList()->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);
		m_GfxDevice->GetDxrCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_Resource.Get()));
	}

	ID3D12Resource* GfxTopLevelAccelerationStructureDX12::GetResource() const
	{
		return m_Resource.Get();
	}

	const GfxHandleDX12& GfxTopLevelAccelerationStructureDX12::GetShaderResourceView() const
	{
		return m_ShaderResourceView;
	}
}
