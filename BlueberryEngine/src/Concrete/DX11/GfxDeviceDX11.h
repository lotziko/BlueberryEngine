#pragma once

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	class GfxTextureDX11;
	class GfxConstantBufferDX11;

	class GfxDeviceDX11 final : public GfxDevice
	{
	public:
		GfxDeviceDX11();
		~GfxDeviceDX11() = default;

		virtual bool Initialize(int width, int height, void* data) final;

		virtual void ClearColor(const Color& color) const final;
		virtual void SwapBuffers() const final;

		virtual void SetViewport(int x, int y, int width, int height) final;
		virtual void ResizeBackbuffer(int width, int height) final;

		virtual bool CreateShader(void* vertexData, void* pixelData, GfxShader*& shader) final;
		virtual bool CreateVertexBuffer(const VertexLayout& layout, const UINT& vertexCount, GfxVertexBuffer*& buffer) final;
		virtual bool CreateIndexBuffer(const UINT& indexCount, GfxIndexBuffer*& buffer) final;
		virtual bool CreateConstantBuffer(const UINT& byteCount, GfxConstantBuffer*& buffer) final;
		virtual bool CreateTexture(const TextureProperties& properties, GfxTexture*& texture) const final;
		virtual bool CreateImGuiRenderer(ImGuiRenderer*& renderer) const final;
		
		virtual void SetRenderTarget(GfxTexture* renderTexture) final;
		virtual void SetGlobalConstantBuffer(const std::size_t& id, GfxConstantBuffer* buffer) final;
		virtual void SetGlobalTexture(const std::size_t& id, GfxTexture* texture) final;
		virtual void Draw(const GfxDrawingOperation& operation) const final;

		virtual Matrix GetGPUMatrix(const Matrix& viewProjection) const final;
	private:
		bool InitializeDirectX(HWND hwnd, int width, int height);

		HWND m_Hwnd;

		ComPtr<ID3D11Device> m_Device;
		ComPtr<ID3D11DeviceContext> m_DeviceContext;
		ComPtr<IDXGISwapChain> m_SwapChain;
		ComPtr<ID3D11RenderTargetView> m_RenderTargetView;
		ComPtr<ID3D11RasterizerState> m_RasterizerState;
		ComPtr<ID3D11DepthStencilState> m_DepthStencilState;

		GfxTextureDX11* m_BindedRenderTarget;
		std::map<std::size_t, GfxConstantBufferDX11*> m_BindedConstantBuffers;
		std::map<std::size_t, GfxTextureDX11*> m_BindedTextures;
	};
}