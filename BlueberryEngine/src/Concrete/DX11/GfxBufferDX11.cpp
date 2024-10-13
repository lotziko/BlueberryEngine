#include "bbpch.h"
#include "GfxBufferDX11.h"

namespace Blueberry
{
	GfxVertexBufferDX11::GfxVertexBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	void GfxVertexBufferDX11::SetData(float* data, const UINT& vertexCount)
	{
		D3D11_MAPPED_SUBRESOURCE mappedBuffer;
		ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

		D3D11_MAP mapType = D3D11_MAP_WRITE_DISCARD;

		m_DeviceContext->Map(m_Buffer.Get(), 0, mapType, 0, &mappedBuffer);
		memcpy(mappedBuffer.pData, data, m_Stride * vertexCount);
		m_DeviceContext->Unmap(m_Buffer.Get(), 0);
	}

	bool GfxVertexBufferDX11::Initialize(const VertexLayout& layout, const UINT& vertexCount)
	{
		m_Layout = layout;
		m_Stride = layout.GetSize();

		D3D11_BUFFER_DESC vertexBufferDesc;
		ZeroMemory(&vertexBufferDesc, sizeof(D3D11_BUFFER_DESC));

		vertexBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		vertexBufferDesc.ByteWidth = m_Stride * vertexCount;
		vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		vertexBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		vertexBufferDesc.MiscFlags = 0;
		vertexBufferDesc.StructureByteStride = 0;

		HRESULT hr = m_Device->CreateBuffer(&vertexBufferDesc, nullptr, m_Buffer.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create vertex buffer."));
			return false;
		}

		return true;
	}

	GfxIndexBufferDX11::GfxIndexBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	void GfxIndexBufferDX11::SetData(UINT* data, const UINT& indexCount)
	{
		D3D11_MAPPED_SUBRESOURCE mappedBuffer;
		ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

		D3D11_MAP mapType = D3D11_MAP_WRITE_DISCARD;

		m_DeviceContext->Map(m_Buffer.Get(), 0, mapType, 0, &mappedBuffer);
		memcpy(mappedBuffer.pData, data, sizeof(UINT) * indexCount);
		m_DeviceContext->Unmap(m_Buffer.Get(), 0);
	}

	bool GfxIndexBufferDX11::Initialize(const UINT& indexCount)
	{
		D3D11_BUFFER_DESC indexBufferDesc;
		ZeroMemory(&indexBufferDesc, sizeof(D3D11_BUFFER_DESC));

		indexBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		indexBufferDesc.ByteWidth = sizeof(UINT) * indexCount;
		indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
		indexBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		indexBufferDesc.MiscFlags = 0;
		indexBufferDesc.StructureByteStride = 0;

		HRESULT hr = m_Device->CreateBuffer(&indexBufferDesc, nullptr, m_Buffer.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create index buffer."));
			return false;
		}

		return true;
	}

	GfxConstantBufferDX11::GfxConstantBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	void GfxConstantBufferDX11::SetData(char* data, const UINT& byteCount)
	{
		D3D11_MAPPED_SUBRESOURCE mappedBuffer;
		ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

		D3D11_MAP mapType = D3D11_MAP_WRITE_DISCARD;

		m_DeviceContext->Map(m_Buffer.Get(), 0, mapType, 0, &mappedBuffer);
		memcpy(mappedBuffer.pData, data, byteCount);
		m_DeviceContext->Unmap(m_Buffer.Get(), 0);
	}

	bool GfxConstantBufferDX11::Initialize(const UINT& byteCount)
	{
		D3D11_BUFFER_DESC constantBufferDesc;
		ZeroMemory(&constantBufferDesc, sizeof(D3D11_BUFFER_DESC));

		constantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		constantBufferDesc.ByteWidth = byteCount % 16 > 0 ? ((byteCount / 16) + 1) * 16 : byteCount;
		constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		constantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		constantBufferDesc.MiscFlags = 0;
		constantBufferDesc.StructureByteStride = 0;

		HRESULT hr = m_Device->CreateBuffer(&constantBufferDesc, nullptr, m_Buffer.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create constant buffer."));
			return false;
		}

		return true;
	}

	GfxStructuredBufferDX11::GfxStructuredBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	void GfxStructuredBufferDX11::SetData(char* data, const UINT& elementCount)
	{
		D3D11_BOX dst;
		dst.left = 0;
		dst.top = 0;
		dst.right = elementCount * m_ElementSize;
		dst.bottom = 1;
		dst.front = 0;
		dst.back = 1;

		m_DeviceContext->UpdateSubresource(m_Buffer.Get(), 0, &dst, data, 0, 0);
	}

	bool GfxStructuredBufferDX11::Initialize(const UINT& elementCount, const UINT& elementSize)
	{
		D3D11_BUFFER_DESC structuredBufferDesc;
		ZeroMemory(&structuredBufferDesc, sizeof(D3D11_BUFFER_DESC));

		UINT byteCount = elementCount * elementSize;
		structuredBufferDesc.Usage = D3D11_USAGE_DEFAULT;
		structuredBufferDesc.ByteWidth = byteCount % 16 > 0 ? ((byteCount / 16) + 1) * 16 : byteCount;
		structuredBufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		structuredBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;//0;
		structuredBufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		structuredBufferDesc.StructureByteStride = elementSize;

		HRESULT hr = m_Device->CreateBuffer(&structuredBufferDesc, nullptr, m_Buffer.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create structured buffer."));
			return false;
		}

		D3D11_SHADER_RESOURCE_VIEW_DESC resourceViewDesc;
		resourceViewDesc.Format = DXGI_FORMAT_UNKNOWN;
		resourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		resourceViewDesc.Buffer.FirstElement = 0;
		resourceViewDesc.Buffer.NumElements = elementCount;

		hr = m_Device->CreateShaderResourceView(m_Buffer.Get(), &resourceViewDesc, m_ShaderResourceView.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create shader resource view."));
			return false;
		}

		m_ElementSize = elementSize;
		return true;
	}

	GfxComputeBufferDX11::GfxComputeBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	void GfxComputeBufferDX11::GetData(char* data, const UINT& byteCount)
	{

	}

	void GfxComputeBufferDX11::SetData(char* data, const UINT& byteCount)
	{

	}

	bool GfxComputeBufferDX11::Initialize(const UINT& elementCount, const UINT& elementSize)
	{
		UINT byteCount = elementCount * elementSize;
		D3D11_BUFFER_DESC computeBufferDesc;
		ZeroMemory(&computeBufferDesc, sizeof(D3D11_BUFFER_DESC));

		computeBufferDesc.ByteWidth = byteCount % 16 > 0 ? ((byteCount / 16) + 1) * 16 : byteCount;
		computeBufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
		computeBufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		computeBufferDesc.StructureByteStride = elementSize;

		HRESULT hr = m_Device->CreateBuffer(&computeBufferDesc, nullptr, m_Buffer.GetAddressOf());
		if (FAILED(hr))
		{
			BB_ERROR(WindowsHelper::GetErrorMessage(hr, "Failed to create compute buffer."));
			return false;
		}

		D3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessViewDesc;
		ZeroMemory(&unorderedAccessViewDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));

		unorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		unorderedAccessViewDesc.Buffer.FirstElement = 0;
		unorderedAccessViewDesc.Format = DXGI_FORMAT_UNKNOWN;
		unorderedAccessViewDesc.Buffer.NumElements = elementCount;

		return true;
	}
}