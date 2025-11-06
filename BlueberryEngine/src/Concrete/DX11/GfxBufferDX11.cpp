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

	void GfxBufferDX11::SetData(const void* data, const uint32_t& size)
	{
		D3D11_MAPPED_SUBRESOURCE mappedBuffer;
		ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

		m_DeviceContext->Map(m_Buffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedBuffer);
		memcpy(mappedBuffer.pData, data, size);
		m_DeviceContext->Unmap(m_Buffer.Get(), 0);
	}

	const uint32_t& GfxBufferDX11::GetElementSize()
	{
		return m_ElementSize;
	}

	UINT GetBindFlags(const BufferType& type)
	{
		switch (type)
		{
		case BufferType::Vertex:
			return D3D11_BIND_VERTEX_BUFFER;
		case BufferType::Index:
			return D3D11_BIND_INDEX_BUFFER;
		case BufferType::Raw:
			return D3D11_BIND_SHADER_RESOURCE;
		case BufferType::Structured:
			return D3D11_BIND_SHADER_RESOURCE;
		case BufferType::Constant:
		default:
			return D3D11_BIND_CONSTANT_BUFFER;
		}
	}

	bool GfxBufferDX11::Initialize(D3D11_SUBRESOURCE_DATA* subresourceData, const BufferProperties& properties)
	{
		m_ElementCount = properties.elementCount;
		m_ElementSize = properties.elementSize;
		m_Type = properties.type;
		uint32_t byteCount = m_ElementCount * m_ElementSize;
		
		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));

		bufferDesc.Usage = properties.isWritable ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT;
		bufferDesc.ByteWidth = properties.type == BufferType::Structured && byteCount % 16 > 0 ? ((byteCount / 16) + 1) * 16 : byteCount;
		bufferDesc.BindFlags = GetBindFlags(properties.type) | (properties.isUnorderedAccess ? D3D11_BIND_UNORDERED_ACCESS : 0);
		bufferDesc.CPUAccessFlags = properties.isWritable ? D3D11_CPU_ACCESS_WRITE : 0;
		
		if (properties.type == BufferType::Structured)
		{
			bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			bufferDesc.StructureByteStride = properties.elementSize;
		}
		else
		{
			bufferDesc.MiscFlags = 0;
			bufferDesc.StructureByteStride = 0;
		}

		HRESULT hr = m_Device->CreateBuffer(&bufferDesc, subresourceData, m_Buffer.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create buffer."));
			return false;
		}

		if (properties.type == BufferType::Structured)
		{
			D3D11_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;
			resourceViewDesc.Format = DXGI_FORMAT_UNKNOWN;
			resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
			resourceViewDesc.Buffer.FirstElement = 0;
			resourceViewDesc.Buffer.NumElements = m_ElementCount;

			hr = m_Device->CreateShaderResourceView(m_Buffer.Get(), &resourceViewDesc, m_ShaderResourceView.GetAddressOf());
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
				return false;
			}
		}

		if (properties.isUnorderedAccess)
		{
			D3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessViewDesc;
			ZeroMemory(&unorderedAccessViewDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));

			unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			unorderedAccessViewDesc.Buffer.FirstElement = 0;
			unorderedAccessViewDesc.Format = static_cast<DXGI_FORMAT>(properties.format);
			unorderedAccessViewDesc.Buffer.NumElements = m_ElementCount;

			hr = m_Device->CreateUnorderedAccessView(m_Buffer.Get(), &unorderedAccessViewDesc, &m_UnorderedAccessView);
			if (FAILED(hr))
			{
				BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create unordered access view."));
				return false;
			}
		}

		if (properties.isReadable)
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