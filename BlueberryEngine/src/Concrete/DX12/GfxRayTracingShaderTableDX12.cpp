#include "GfxRayTracingShaderTableDX12.h"

#include "..\Windows\WindowsHelper.h"
#include "GfxDeviceDX12.h"

namespace Blueberry
{
	GfxRayTracingShaderTableDX12::GfxRayTracingShaderTableDX12(GfxDeviceDX12* device) : m_GfxDevice(device)
	{
	}

	GfxRayTracingShaderTableDX12::GfxRayTracingShaderTableDX12(GfxDeviceDX12* device, size_t initialSize) : m_GfxDevice(device)
	{
		m_Data.reserve(initialSize);
		m_Size = initialSize;
	}

	void GfxRayTracingShaderTableDX12::Clear()
	{
		m_Data.clear();
		m_RecordCount = 0;
	}

	void GfxRayTracingShaderTableDX12::AddRecord(void* shaderIdentifier, size_t shaderIndentifierSize, void* rootArguments, size_t rootArgumentsSize)
	{
		size_t size = m_Data.size();
		m_Data.resize(size + shaderIndentifierSize + rootArgumentsSize);
		uint8_t* ptr = m_Data.data() + size;
		memcpy(ptr, shaderIdentifier, shaderIndentifierSize);
		if (rootArgumentsSize > 0)
		{
			ptr += shaderIndentifierSize;
			memcpy(ptr, rootArguments, rootArgumentsSize);
		}
		++m_RecordCount;
	}

	void GfxRayTracingShaderTableDX12::Build()
	{
		if (m_Data.size() > m_Size)
		{
			m_GfxDevice->Release(m_Resource);
			m_Resource = nullptr;
			m_Size = m_Data.size();
		}
		if (m_Resource.Get() == nullptr)
		{
			D3D12_RESOURCE_DESC resourceDesc = {};
			resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
			resourceDesc.Width = m_Size;
			resourceDesc.Height = 1;
			resourceDesc.DepthOrArraySize = 1;
			resourceDesc.MipLevels = 1;
			resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
			resourceDesc.SampleDesc.Count = 1;
			resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
			resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

			HRESULT hr = m_GfxDevice->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_Resource));
			
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader table buffer."));
				return;
			}

			m_GfxDevice->GetCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));
		}
		else
		{
			m_GfxDevice->GetCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST));
		}
		m_GfxDevice->GetUploadBuffer().UploadBuffer(m_Resource.Get(), m_Data.data(), m_Data.size(), D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
		m_GfxDevice->GetCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
	}

	ID3D12Resource* GfxRayTracingShaderTableDX12::GetResource() const
	{
		return m_Resource.Get();
	}

	size_t GfxRayTracingShaderTableDX12::GetSize() const
	{
		return m_Data.size();
	}

	size_t GfxRayTracingShaderTableDX12::GetRecordCount() const
	{
		return m_RecordCount;
	}
}