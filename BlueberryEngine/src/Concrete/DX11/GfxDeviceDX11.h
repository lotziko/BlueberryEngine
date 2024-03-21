#pragma once

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	class GfxTextureDX11;
	class GfxConstantBufferDX11;

	class GfxDeviceDX11 final : public GfxDevice
	{
	public:
		GfxDeviceDX11() = default;
		~GfxDeviceDX11() = default;

	protected:
		virtual bool InitializeImpl(int width, int height, void* data) final;

		virtual void ClearColorImpl(const Color& color) const final;
		virtual void ClearDepthImpl(const float& depth) const final;
		virtual void SwapBuffersImpl() const final;

		virtual void SetViewportImpl(int x, int y, int width, int height) final;
		virtual void ResizeBackbufferImpl(int width, int height) final;

		virtual bool CreateShaderImpl(void* vertexData, void* pixelData, GfxShader*& shader) final;
		virtual bool CreateComputeShaderImpl(void* computeData, GfxComputeShader*& shader) final;
		virtual bool CreateVertexBufferImpl(const VertexLayout& layout, const UINT& vertexCount, GfxVertexBuffer*& buffer) final;
		virtual bool CreateIndexBufferImpl(const UINT& indexCount, GfxIndexBuffer*& buffer) final;
		virtual bool CreateConstantBufferImpl(const UINT& byteCount, GfxConstantBuffer*& buffer) final;
		virtual bool CreateComputeBufferImpl(const UINT& elementCount, const UINT& elementSize, GfxComputeBuffer*& buffer) final;
		virtual bool CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) const final;
		virtual bool CreateImGuiRendererImpl(ImGuiRenderer*& renderer) const final;

		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) const final;
		
		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture) final;
		virtual void SetGlobalConstantBufferImpl(const std::size_t& id, GfxConstantBuffer* buffer) final;
		virtual void SetGlobalTextureImpl(const std::size_t& id, GfxTexture* texture) final;
		virtual void DrawImpl(const GfxDrawingOperation& operation) final;

		virtual void DispatchImpl(GfxComputeShader*& shader, const UINT& threadGroupsX, const UINT& threadGroupsY, const UINT& threadGroupsZ) const final;

		virtual Matrix GetGPUMatrixImpl(const Matrix& viewProjection) const final;

	private:
		bool InitializeDirectX(HWND hwnd, int width, int height);

		void SetCullMode(const CullMode& mode);
		void SetSurfaceType(const SurfaceType& type);

		HWND m_Hwnd;

		ComPtr<ID3D11Device> m_Device;
		ComPtr<ID3D11DeviceContext> m_DeviceContext;
		ComPtr<IDXGISwapChain> m_SwapChain;
		ComPtr<ID3D11RenderTargetView> m_RenderTargetView;

		ComPtr<ID3D11RasterizerState> m_CullNoneRasterizerState;
		ComPtr<ID3D11RasterizerState> m_CullFrontRasterizerState;

		ComPtr<ID3D11DepthStencilState> m_OpaqueDepthStencilState;
		ComPtr<ID3D11BlendState> m_OpaqueBlendState;

		ComPtr<ID3D11DepthStencilState> m_TransparentDepthStencilState;
		ComPtr<ID3D11BlendState> m_TransparentBlendState;

		GfxTextureDX11* m_BindedRenderTarget;
		GfxTextureDX11* m_BindedDepthStencil;
		std::map<std::size_t, GfxConstantBufferDX11*> m_BindedConstantBuffers;
		std::map<std::size_t, GfxTextureDX11*> m_BindedTextures;

		CullMode m_CullMode = (CullMode)-1;
		SurfaceType m_SurfaceType = (SurfaceType)-1;
	};
}