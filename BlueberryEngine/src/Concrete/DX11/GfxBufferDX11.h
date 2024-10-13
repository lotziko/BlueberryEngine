#pragma once

#include "Blueberry\Graphics\GfxBuffer.h"
#include "Blueberry\Graphics\VertexLayout.h"

namespace Blueberry
{
	class GfxVertexBufferDX11 final : public GfxVertexBuffer
	{
	public:
		GfxVertexBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxVertexBufferDX11() final = default;
		virtual void SetData(float* data, const UINT& vertexCount) final;

		bool Initialize(const VertexLayout& layout, const UINT& vertexCount);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		VertexLayout m_Layout;
		UINT m_Stride;
		UINT m_Offset = 0;

		friend class GfxDeviceDX11;
	};

	class GfxIndexBufferDX11 final : public GfxIndexBuffer
	{
	public:
		GfxIndexBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxIndexBufferDX11() = default;
		virtual void SetData(UINT* data, const UINT& indexCount) final;

		bool Initialize(const UINT& indexCount);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
	};

	class GfxConstantBufferDX11 final : public GfxConstantBuffer
	{
	public:
		GfxConstantBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxConstantBufferDX11() = default;
		virtual void SetData(char* data, const UINT& byteCount) final;

		bool Initialize(const UINT& byteCount);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
	};

	class GfxStructuredBufferDX11 final : public GfxStructuredBuffer
	{
	public:
		GfxStructuredBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxStructuredBufferDX11() = default;
		virtual void SetData(char* data, const UINT& elementCount) final;

		bool Initialize(const UINT& elementCount, const UINT& elementSize);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;
		ComPtr<ID3D11ShaderResourceView> m_ShaderResourceView = nullptr;
		UINT m_ElementSize;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
	};

	class GfxComputeBufferDX11 final : public GfxComputeBuffer
	{
	public:
		GfxComputeBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxComputeBufferDX11() = default;
		virtual void GetData(char* data, const UINT& byteCount) final;
		virtual void SetData(char* data, const UINT& byteCount) final;

		bool Initialize(const UINT& elementCount, const UINT& elementSize);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;
		ComPtr<ID3D11UnorderedAccessView> m_UnorderedAccessView = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
	};
}