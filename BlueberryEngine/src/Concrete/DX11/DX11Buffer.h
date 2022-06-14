#pragma once

#include "Blueberry\Graphics\Buffer.h"
#include "Blueberry\Graphics\VertexLayout.h"

namespace Blueberry
{
	class DX11VertexBuffer final : public VertexBuffer
	{
	public:
		DX11VertexBuffer(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~DX11VertexBuffer() final = default;
		virtual void Bind() final;
		virtual void SetData(float* data, const UINT& vertexCount) final;

		bool Initialize(const VertexLayout& layout, const UINT& vertexCount);
	private:
		ComRef<ID3D11Buffer> m_Buffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;

		VertexLayout m_Layout;
		UINT m_Stride;
		UINT m_Offset = 0;
	};

	class DX11IndexBuffer final : public IndexBuffer
	{
	public:
		DX11IndexBuffer(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~DX11IndexBuffer();
		virtual void Bind() final;
		virtual void SetData(UINT* data, const UINT& indexCount) final;

		bool Initialize(const UINT& indexCount);
	private:
		ComRef<ID3D11Buffer> m_Buffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;
	};

	class DX11ConstantBuffer final : public ConstantBuffer
	{
	public:
		DX11ConstantBuffer(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
		virtual ~DX11ConstantBuffer();
		virtual void Bind() final;
		virtual void SetData(char* data, const UINT& byteCount) final;

		bool Initialize(const UINT& byteCount);
	private:
		ComRef<ID3D11Buffer> m_Buffer = nullptr;

		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;
	};
}