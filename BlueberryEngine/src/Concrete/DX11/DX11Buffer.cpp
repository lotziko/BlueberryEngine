#include "bbpch.h"
#include "DX11Buffer.h"

DX11VertexBuffer::DX11VertexBuffer(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
{
}

void DX11VertexBuffer::Bind()
{
	m_DeviceContext->IASetVertexBuffers(0, 1, m_Buffer.GetAddressOf(), &m_Stride, &m_Offset);
}

void DX11VertexBuffer::SetData(float* data, const UINT& vertexCount)
{
	D3D11_MAPPED_SUBRESOURCE mappedBuffer;
	ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

	D3D11_MAP mapType = D3D11_MAP_WRITE_DISCARD;
	
	m_DeviceContext->Map(m_Buffer.Get(), 0, mapType, 0, &mappedBuffer);
	memcpy(mappedBuffer.pData, data, m_Stride * vertexCount);
	m_DeviceContext->Unmap(m_Buffer.Get(), 0);
}

bool DX11VertexBuffer::Initialize(const VertexLayout& layout, const UINT& vertexCount)
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

DX11IndexBuffer::DX11IndexBuffer(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
{
}

DX11IndexBuffer::~DX11IndexBuffer()
{
}

void DX11IndexBuffer::Bind()
{
	m_DeviceContext->IASetIndexBuffer(m_Buffer.Get(), DXGI_FORMAT_R32_UINT, 0);
}

void DX11IndexBuffer::SetData(UINT* data, const UINT& indexCount)
{
	D3D11_MAPPED_SUBRESOURCE mappedBuffer;
	ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

	D3D11_MAP mapType = D3D11_MAP_WRITE_DISCARD;

	m_DeviceContext->Map(m_Buffer.Get(), 0, mapType, 0, &mappedBuffer);
	memcpy(mappedBuffer.pData, data, sizeof(UINT) * indexCount);
	m_DeviceContext->Unmap(m_Buffer.Get(), 0);
}

bool DX11IndexBuffer::Initialize(const UINT& indexCount)
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

DX11ConstantBuffer::DX11ConstantBuffer(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
{
}

DX11ConstantBuffer::~DX11ConstantBuffer()
{
}

void DX11ConstantBuffer::Bind()
{
	m_DeviceContext->VSSetConstantBuffers(0, 1, m_Buffer.GetAddressOf());
}

void DX11ConstantBuffer::SetData(char* data, const UINT& byteCount)
{
	D3D11_MAPPED_SUBRESOURCE mappedBuffer;
	ZeroMemory(&mappedBuffer, sizeof(D3D11_MAPPED_SUBRESOURCE));

	D3D11_MAP mapType = D3D11_MAP_WRITE_DISCARD;

	m_DeviceContext->Map(m_Buffer.Get(), 0, mapType, 0, &mappedBuffer);
	memcpy(mappedBuffer.pData, data, byteCount);
	m_DeviceContext->Unmap(m_Buffer.Get(), 0);
}

bool DX11ConstantBuffer::Initialize(const UINT& byteCount)
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
