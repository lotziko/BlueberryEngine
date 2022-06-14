#pragma once

#include "Blueberry\Graphics\GraphicsDevice.h"

namespace Blueberry
{
	class DX11GraphicsDevice final : public GraphicsDevice
	{
	public:
		DX11GraphicsDevice();
		~DX11GraphicsDevice() = default;

		virtual bool Initialize(int width, int height, void* data) final;

		virtual void ClearColor(const Color& color) const final;
		virtual void SwapBuffers() const final;

		virtual void SetViewport(int x, int y, int width, int height) final;
		virtual void ResizeBackbuffer(int width, int height) final;

		virtual bool CreateShader(const std::wstring& vertexShaderPath, const std::wstring& pixelShaderPath, Ref<Shader>& shader) final;
		virtual bool CreateVertexBuffer(const VertexLayout& layout, const UINT& vertexCount, Ref<VertexBuffer>& buffer) final;
		virtual bool CreateIndexBuffer(const UINT& indexCount, Ref<IndexBuffer>& buffer) final;
		virtual bool CreateConstantBuffer(const UINT& byteCount, Ref<ConstantBuffer>& buffer) final;
		virtual bool CreateTexture(const std::string& path, Ref<Texture>& texture) const final;
		virtual bool CreateImGuiRenderer(Ref<ImGuiRenderer>& renderer) const final;

		virtual void Draw(const int& vertices) const final;
		virtual void DrawIndexed(const int& indices) const final;

		virtual Matrix GetGPUMatrix(const Matrix& viewProjection) const final;
	private:
		bool InitializeDirectX(HWND hwnd, int width, int height);

		HWND m_Hwnd;

		ComRef<ID3D11Device> m_Device;
		ComRef<ID3D11DeviceContext> m_DeviceContext;
		ComRef<IDXGISwapChain> m_SwapChain;
		ComRef<ID3D11RenderTargetView> m_RenderTargetView;
		ComRef<ID3D11RasterizerState> m_RasterizerState;
		ComRef<ID3D11DepthStencilState> m_DepthStencilState;
	};
}