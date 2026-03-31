#include "GfxBufferDX11.h"

#include "..\Windows\WindowsHelper.h"

namespace Blueberry
{
	GfxPointerCacheDX11<GfxBufferDX11> GfxBufferDX11::s_PointerCache = {};

	GfxBufferDX11::GfxBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
		m_Index = s_PointerCache.Allocate(this);
	}

	GfxBufferDX11::~GfxBufferDX11()
	{
		s_PointerCache.Deallocate(m_Index);
	}

	bool GfxBufferDX11::Initialize(const BufferProperties& properties)
	{
		if (properties.dataSize > 0)
		{
			D3D11_SUBRESOURCE_DATA subresourceData;
			subresourceData.pSysMem = properties.data;
			subresourceData.SysMemPitch = static_cast<UINT>(properties.dataSize);
			subresourceData.SysMemSlicePitch = 0;
			return Initialize(&subresourceData, properties);
		}
		else
		{
			return Initialize(nullptr, properties);
		}
	}

	void GfxBufferDX11::GetData(void* data)
	{
		m_DeviceContext->CopyResource(m_StagingBuffer.Get(), m_Buffer.Get());

		D3D11_MAPPED_SUBRESOURCE mappedBuffer;
		ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

		HRESULT hr = m_DeviceContext->Map(m_StagingBuffer.Get(), 0, D3D11_MAP_READ, 0, &mappedBuffer);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to get buffer data."));
			return;
		}
		memcpy(data, mappedBuffer.pData, m_ElementCount * m_ElementSize);
		m_DeviceContext->Unmap(m_StagingBuffer.Get(), 0);
	}

	void GfxBufferDX11::SetData(const void* data, size_t size)
	{
		D3D11_MAPPED_SUBRESOURCE mappedBuffer;
		ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

		HRESULT hr = m_DeviceContext->Map(m_Buffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedBuffer);
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to write to the buffer."));
			return;
		}
		memcpy(mappedBuffer.pData, data, size);
		m_DeviceContext->Unmap(m_Buffer.Get(), 0);
	}

	uint32_t GfxBufferDX11::GetElementSize() const
	{
		return m_ElementSize;
	}

	uint32_t GfxBufferDX11::GetElementCount() const
	{
		return m_ElementCount;
	}

	bool HasFlag(BufferUsageFlags usageFlags, BufferUsageFlags flag)
	{
		return (usageFlags & flag) != BufferUsageFlags::None;
	}

	UINT GetBindFlags(const BufferUsageFlags& usageFlags)
	{
		UINT bindFlags = 0;
		if (HasFlag(usageFlags, BufferUsageFlags::VertexBuffer))
		{
			bindFlags |= D3D11_BIND_VERTEX_BUFFER;
		}
		else if (HasFlag(usageFlags, BufferUsageFlags::IndexBuffer))
		{
			bindFlags |= D3D11_BIND_INDEX_BUFFER;
		}
		else if (HasFlag(usageFlags, BufferUsageFlags::ConstantBuffer))
		{
			bindFlags |= D3D11_BIND_CONSTANT_BUFFER;
		}
		if (HasFlag(usageFlags, BufferUsageFlags::ShaderResource))
		{
			bindFlags |= D3D11_BIND_SHADER_RESOURCE;
		}
		if (HasFlag(usageFlags, BufferUsageFlags::UnorderedAccess))
		{
			bindFlags |= D3D11_BIND_UNORDERED_ACCESS;
		}
		return bindFlags;
	}

	bool GfxBufferDX11::Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, const BufferProperties& properties)
	{
		m_ElementCount = properties.elementCount;
		m_ElementSize = properties.elementSize;
		m_IsConstant = HasFlag(properties.usageFlags, BufferUsageFlags::ConstantBuffer);
		uint32_t byteCount = m_ElementCount * m_ElementSize;

		bool isWritable = m_IsConstant || HasFlag(properties.usageFlags, BufferUsageFlags::CPUWritable);
		bool isRaw = HasFlag(properties.usageFlags, BufferUsageFlags::ByteAdressBuffer);

		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));

		bufferDesc.Usage = isWritable ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT;
		bufferDesc.ByteWidth = HasFlag(properties.usageFlags, BufferUsageFlags::StructuredBuffer) && byteCount % 16 > 0 ? ((byteCount / 16) + 1) * 16 : byteCount;
		bufferDesc.BindFlags = GetBindFlags(properties.usageFlags);
		bufferDesc.CPUAccessFlags = isWritable ? D3D11_CPU_ACCESS_WRITE : 0;
		
		if (HasFlag(properties.usageFlags, BufferUsageFlags::StructuredBuffer))
		{
			bufferDesc.MiscFlags |= D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			bufferDesc.StructureByteStride = properties.elementSize;
		}
		if (isRaw)
		{
			bufferDesc.MiscFlags |= D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
		}

		HRESULT hr = m_Device->CreateBuffer(&bufferDesc, subresourceData, m_Buffer.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create buffer."));
			return false;
		}

		if (HasFlag(properties.usageFlags, BufferUsageFlags::ShaderResource))
		{
			D3D11_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;
			
			if (isRaw)
			{
				resourceViewDesc.Format = DXGI_FORMAT_R32_TYPELESS;
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
				resourceViewDesc.BufferEx.FirstElement = 0;
				resourceViewDesc.BufferEx.NumElements = byteCount / sizeof(uint32_t);
				resourceViewDesc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
			}
			else
			{
				resourceViewDesc.Format = static_cast<DXGI_FORMAT>(properties.format);
				resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
				resourceViewDesc.Buffer.FirstElement = 0;
				resourceViewDesc.Buffer.NumElements = m_ElementCount;
			}

			hr = m_Device->CreateShaderResourceView(m_Buffer.Get(), &resourceViewDesc, m_ShaderResourceView.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
				return false;
			}
		}
		
		if (HasFlag(properties.usageFlags, BufferUsageFlags::UnorderedAccess))
		{
			D3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessViewDesc;
			ZeroMemory(&unorderedAccessViewDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));

			unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;

			if (isRaw)
			{
				unorderedAccessViewDesc.Format = DXGI_FORMAT_R32_TYPELESS;
				unorderedAccessViewDesc.Buffer.FirstElement = 0;
				unorderedAccessViewDesc.Buffer.NumElements = byteCount / sizeof(uint32_t);
				unorderedAccessViewDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
			}
			else
			{
				unorderedAccessViewDesc.Format = static_cast<DXGI_FORMAT>(properties.format);
				unorderedAccessViewDesc.Buffer.FirstElement = 0;
				unorderedAccessViewDesc.Buffer.NumElements = m_ElementCount;
			}

			hr = m_Device->CreateUnorderedAccessView(m_Buffer.Get(), &unorderedAccessViewDesc, &m_UnorderedAccessView);
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create unordered access view."));
				return false;
			}
		}

		if (HasFlag(properties.usageFlags, BufferUsageFlags::CPUReadable))
		{
			D3D11_BUFFER_DESC stagingBufferDesc;
			ZeroMemory(&stagingBufferDesc, sizeof(D3D11_BUFFER_DESC));

			stagingBufferDesc.Usage = D3D11_USAGE_STAGING;
			stagingBufferDesc.ByteWidth = bufferDesc.ByteWidth;
			stagingBufferDesc.BindFlags = 0;
			stagingBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

			HRESULT hr = m_Device->CreateBuffer(&stagingBufferDesc, nullptr, m_StagingBuffer.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create staging buffer."));
				return false;
			}
		}

		return true;
	}
}