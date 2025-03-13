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
		virtual void SetData(float* data, const uint32_t& vertexCount) final;

		bool Initialize(const VertexLayout& layout, const uint32_t& vertexCount);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		VertexLayout m_Layout;
		uint32_t m_Stride;
		uint32_t m_Offset = 0;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};

	class GfxIndexBufferDX11 final : public GfxIndexBuffer
	{
	public:
		GfxIndexBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxIndexBufferDX11() = default;
		virtual void SetData(uint32_t* data, const uint32_t& indexCount) final;

		bool Initialize(const uint32_t& indexCount);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};

	class GfxConstantBufferDX11 final : public GfxConstantBuffer
	{
	public:
		GfxConstantBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxConstantBufferDX11() = default;
		virtual void SetData(char* data, const uint32_t& byteCount) final;

		bool Initialize(const uint32_t& byteCount);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};

	class GfxStructuredBufferDX11 final : public GfxStructuredBuffer
	{
	public:
		GfxStructuredBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxStructuredBufferDX11() = default;
		virtual void SetData(char* data, const uint32_t& elementCount) final;

		bool Initialize(const uint32_t& elementCount, const uint32_t& elementSize);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;
		ComPtr<ID3D11ShaderResourceView> m_ShaderResourceView = nullptr;
		uint32_t m_ElementSize;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
		friend class GfxRenderStateCacheDX11;
	};

	class GfxComputeBufferDX11 final : public GfxComputeBuffer
	{
	public:
		GfxComputeBufferDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~GfxComputeBufferDX11() = default;
		virtual void GetData(char* data, const uint32_t& byteCount) final;
		virtual void SetData(char* data, const uint32_t& byteCount) final;

		bool Initialize(const uint32_t& elementCount, const uint32_t& elementSize);
	private:
		ComPtr<ID3D11Buffer> m_Buffer = nullptr;
		ComPtr<ID3D11UnorderedAccessView> m_UnorderedAccessView = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		friend class GfxDeviceDX11;
	};
}