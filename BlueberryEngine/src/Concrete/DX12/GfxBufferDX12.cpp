#include "GfxBufferDX12.h"

#include "..\Windows\WindowsHelper.h"
#include "GfxDeviceDX12.h"

namespace Blueberry
{
	GfxPointerCache<GfxBufferDX12> GfxBufferDX12::s_PointerCache = {};

	GfxBufferDX12::GfxBufferDX12(GfxDeviceDX12* device) : m_GfxDevice(device), m_Device(device->GetDevice())
	{
		m_Index = s_PointerCache.Allocate(this);
	}

	GfxBufferDX12::~GfxBufferDX12()
	{
		s_PointerCache.Deallocate(m_Index);
		m_ShaderResourceView.Free();
		m_UnorderedAccessView.Free();
		m_ConstantBufferView.Free();
		m_GfxDevice->Release(m_Resource);
	}

	bool GfxBufferDX12::Initialize(const BufferProperties& properties)
	{
		if (properties.dataSize > 0)
		{
			D3D12_SUBRESOURCE_DATA subresourceData = {};
			subresourceData.pData = properties.data;
			subresourceData.RowPitch = static_cast<UINT>(properties.dataSize);
			subresourceData.SlicePitch = 0;
			return Initialize(&subresourceData, properties);
		}
		else
		{
			return Initialize(nullptr, properties);
		}
	}

	void GfxBufferDX12::GetData(void* data)
	{
		if (data != nullptr)
		{
			SetState(D3D12_RESOURCE_STATE_COPY_SOURCE);
			m_GfxDevice->GetReadbackBuffer().ReadBuffer(m_Resource.Get(), data, m_ElementCount * m_ElementSize);
		}
	}

	void GfxBufferDX12::SetData(const void* data, size_t size)
	{
		if (data != nullptr && size > 0)
		{
			SetState(D3D12_RESOURCE_STATE_COPY_DEST);
			m_GfxDevice->GetUploadBuffer().UploadBuffer(m_Resource.Get(), data, size, m_IsConstant ? 256ull : 4ull);
		}
	}

	ID3D12Resource* GfxBufferDX12::GetResource()
	{
		return m_Resource.Get();
	}

	const GfxHandleDX12& GfxBufferDX12::GetShaderResourceView() const
	{
		return m_ShaderResourceView;
	}

	const GfxHandleDX12& GfxBufferDX12::GetUnorderedAccessView() const
	{
		return m_UnorderedAccessView;
	}

	const GfxHandleDX12& GfxBufferDX12::GetConstantBufferView() const
	{
		return m_ConstantBufferView;
	}

	D3D12_VERTEX_BUFFER_VIEW GfxBufferDX12::GetVertexView()
	{
		D3D12_VERTEX_BUFFER_VIEW vertexBufferView = {};
		vertexBufferView.BufferLocation = m_Resource->GetGPUVirtualAddress();
		vertexBufferView.SizeInBytes = m_ElementCount * m_ElementSize;
		vertexBufferView.StrideInBytes = m_ElementSize;
		return vertexBufferView;
	}

	D3D12_INDEX_BUFFER_VIEW GfxBufferDX12::GetIndexView()
	{
		D3D12_INDEX_BUFFER_VIEW indexBufferView = {};
		indexBufferView.BufferLocation = m_Resource->GetGPUVirtualAddress();
		indexBufferView.SizeInBytes = m_ElementCount * m_ElementSize;
		indexBufferView.Format = DXGI_FORMAT_R32_UINT;
		return indexBufferView;
	}

	D3D12_RESOURCE_STATES GfxBufferDX12::GetState() const
	{
		return m_State;
	}

	void GfxBufferDX12::SetState(D3D12_RESOURCE_STATES state)
	{
		if (m_State != state)
		{
			m_GfxDevice->GetCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), m_State, state));
			m_State = state;
		}
	}

	void GfxBufferDX12::SetUAVState()
	{
		if (m_State == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
		{
			uint64_t generation = m_GfxDevice->GetGeneration();
			if (m_Generation == generation)
			{
				m_GfxDevice->GetCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_Resource.Get()));
			}
			m_Generation = generation;
		}
		else
		{
			m_GfxDevice->GetCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(), m_State, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
			m_State = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		}
	}

	GfxBufferDX12* GfxBufferDX12::Get(uint32_t index)
	{
		return s_PointerCache.Get(index);
	}

	bool GfxBufferDX12::Initialize(D3D12_SUBRESOURCE_DATA* subresourceData, const BufferProperties& properties)
	{
		m_ElementCount = properties.elementCount;
		m_ElementSize = properties.elementSize;
		uint32_t byteCount = m_ElementCount * m_ElementSize;

		bool useSRV = HasFlag(properties.usageFlags, BufferUsageFlags::ShaderResource);
		bool useUAV = HasFlag(properties.usageFlags, BufferUsageFlags::UnorderedAccess);
		bool isConstant = m_IsConstant = HasFlag(properties.usageFlags, BufferUsageFlags::ConstantBuffer);
		bool isWritable = isConstant || HasFlag(properties.usageFlags, BufferUsageFlags::CPUWritable);
		bool isStructured = HasFlag(properties.usageFlags, BufferUsageFlags::StructuredBuffer);
		bool isRaw = HasFlag(properties.usageFlags, BufferUsageFlags::ByteAdressBuffer);
		bool isVertex = HasFlag(properties.usageFlags, BufferUsageFlags::VertexBuffer);
		bool isIndex = HasFlag(properties.usageFlags, BufferUsageFlags::IndexBuffer);

		if (isStructured)
		{
			byteCount = Math::NextDivisableBy(byteCount, 16u);
		}

		if (isConstant)
		{
			byteCount = Math::NextDivisableBy(byteCount, 256u);
		}

		D3D12_RESOURCE_DESC resourceDesc = {};
		resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		resourceDesc.Width = byteCount;
		resourceDesc.Height = 1;
		resourceDesc.DepthOrArraySize = 1;
		resourceDesc.MipLevels = 1;
		resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
		resourceDesc.SampleDesc.Count = 1;
		resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

		if (useUAV)
		{
			resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		}
		
		m_State = D3D12_RESOURCE_STATE_COMMON;
		HRESULT hr = m_Device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &resourceDesc, m_State, nullptr, IID_PPV_ARGS(&m_Resource));

		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create buffer."));
			return false;
		}

		if (subresourceData != nullptr)
		{
			SetState(D3D12_RESOURCE_STATE_COPY_DEST);
			m_GfxDevice->GetUploadBuffer().UploadBuffer(m_Resource.Get(), subresourceData->pData, subresourceData->RowPitch, isConstant ? 256ull : 4ull);
		}

		if (useSRV)
		{
			D3D12_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc = {};
			shaderResourceViewDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
			shaderResourceViewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			shaderResourceViewDesc.Buffer.FirstElement = 0;

			if (isRaw)
			{
				shaderResourceViewDesc.Format = DXGI_FORMAT_R32_TYPELESS;
				shaderResourceViewDesc.Buffer.NumElements = byteCount / sizeof(uint32_t);
				shaderResourceViewDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
			}
			else
			{
				shaderResourceViewDesc.Format = DXGI_FORMAT_UNKNOWN;
				shaderResourceViewDesc.Buffer.NumElements = m_ElementCount;
				shaderResourceViewDesc.Buffer.StructureByteStride = properties.elementSize;
				shaderResourceViewDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
			}

			m_ShaderResourceView = m_GfxDevice->GetCbvSrvUavHeap().AllocatePersistent();
			m_Device->CreateShaderResourceView(m_Resource.Get(), &shaderResourceViewDesc, m_ShaderResourceView.GetCPU());
		}

		if (useUAV)
		{
			D3D12_UNORDERED_ACCESS_VIEW_DESC unorderedAccessViewDesc = {};
			unorderedAccessViewDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			unorderedAccessViewDesc.Buffer.FirstElement = 0;

			if (isRaw)
			{
				unorderedAccessViewDesc.Format = DXGI_FORMAT_R32_TYPELESS;
				unorderedAccessViewDesc.Buffer.NumElements = byteCount / sizeof(uint32_t);
				unorderedAccessViewDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
			}
			else
			{
				unorderedAccessViewDesc.Format = DXGI_FORMAT_UNKNOWN;
				unorderedAccessViewDesc.Buffer.NumElements = m_ElementCount;
				unorderedAccessViewDesc.Buffer.StructureByteStride = properties.elementSize;
				unorderedAccessViewDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
			}

			m_UnorderedAccessView = m_GfxDevice->GetCbvSrvUavHeap().AllocatePersistent();
			m_Device->CreateUnorderedAccessView(m_Resource.Get(), nullptr, &unorderedAccessViewDesc, m_UnorderedAccessView.GetCPU());
		}

		if (isConstant)
		{
			D3D12_CONSTANT_BUFFER_VIEW_DESC constantBufferViewDesc = {};
			constantBufferViewDesc.BufferLocation = m_Resource->GetGPUVirtualAddress();
			constantBufferViewDesc.SizeInBytes = byteCount;

			m_ConstantBufferView = m_GfxDevice->GetCbvSrvUavHeap().AllocatePersistent();
			m_Device->CreateConstantBufferView(&constantBufferViewDesc, m_ConstantBufferView.GetCPU());
		}

		return true;
	}
}
