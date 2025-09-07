#pragma once

#include "Blueberry\Graphics\GfxDevice.h"
#include "Concrete\Windows\ComPtr.h"

#include "GfxRenderStateCacheDX11.h"
#include "GfxInputLayoutCacheDX11.h"
#include "Concrete\DX11\DX11.h"

namespace Blueberry
{
	class GfxTextureDX11;
	class GfxBufferDX11;
	class GfxVertexShaderDX11;
	class GfxGeometryShaderDX11;
	class GfxFragmentShaderDX11;
	struct GfxRenderStateDX11;

	class GfxDeviceDX11 final : public GfxDevice
	{
	public:
		GfxDeviceDX11() = default;
		~GfxDeviceDX11() = default;

	protected:
		virtual bool InitializeImpl(int width, int height, void* data) final;

		virtual void ClearColorImpl(const Color& color) const final;
		virtual void ClearDepthImpl(const float& depth) const final;
		virtual void SwapBuffersImpl() final;

		virtual void SetViewportImpl(int x, int y, int width, int height) final;
		virtual void SetScissorRectImpl(int x, int y, int width, int height) final;
		virtual void ResizeBackbufferImpl(int width, int height) final;

		virtual const uint32_t& GetViewCountImpl() final;
		virtual void SetViewCountImpl(const uint32_t& count) final;
		virtual void SetDepthBiasImpl(const uint32_t& bias, const float& slopeBias) final;

		virtual bool CreateVertexShaderImpl(void* vertexData, GfxVertexShader*& shader) final;
		virtual bool CreateGeometryShaderImpl(void* geometryData, GfxGeometryShader*& shader) final;
		virtual bool CreateFragmentShaderImpl(void* fragmentData, GfxFragmentShader*& shader) final;
		virtual bool CreateComputeShaderImpl(void* computeData, GfxComputeShader*& shader) final;
		virtual bool CreateBufferImpl(const BufferProperties& properties, GfxBuffer*& buffer) final;
		virtual bool CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) const final;

		virtual void CopyImpl(GfxTexture* source, GfxTexture* target) const final;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) const final;

		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture) final;
		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, const uint32_t& slice) final;
		virtual void SetGlobalBufferImpl(const size_t& id, GfxBuffer* buffer) final;
		virtual void SetGlobalTextureImpl(const size_t& id, GfxTexture* texture) final;
		virtual void DrawImpl(const GfxDrawingOperation& operation) final;

		virtual void DispatchImpl(GfxComputeShader* shader, const uint32_t& threadGroupsX, const uint32_t& threadGroupsY, const uint32_t& threadGroupsZ) final;

		virtual Matrix GetGPUMatrixImpl(const Matrix& viewProjection) const final;

	public:
		ID3D11Device* GetDevice();
		ID3D11DeviceContext* GetDeviceContext();
		HWND GetHwnd();

	private:
		bool InitializeDirectX(HWND hwnd, int width, int height);

		void Clear();

		ID3D11RasterizerState* GetRasterizerState(const CullMode& mode);
		ID3D11BlendState* GetBlendState(const BlendMode& blendSrcColor, const BlendMode& blendSrcAlpha, const BlendMode& blendDstColor, const BlendMode& blendDstAlpha);
		ID3D11DepthStencilState* GetDepthStencilState(const ZTest& zTest, const ZWrite& zWrite);

		const uint32_t& GetCRC();

		HWND m_Hwnd;

		ComPtr<ID3D11Device> m_Device;
		ComPtr<ID3D11DeviceContext> m_DeviceContext;
		ComPtr<IDXGISwapChain> m_SwapChain;
		ComPtr<ID3D11RenderTargetView> m_RenderTargetView;

		Dictionary<size_t, ComPtr<ID3D11RasterizerState>> m_RasterizerStates;
		Dictionary<size_t, ComPtr<ID3D11DepthStencilState>> m_DepthStencilStates;
		Dictionary<size_t, ComPtr<ID3D11BlendState>> m_BlendStates;

		GfxTextureDX11* m_BindedRenderTarget;
		GfxTextureDX11* m_BindedDepthStencil;
		List<std::pair<size_t, GfxBufferDX11*>> m_BindedBuffers;
		List<std::pair<size_t, GfxTextureDX11*>> m_BindedTextures;
		ID3D11ShaderResourceView* m_EmptyShaderResourceViews[16];
		ID3D11SamplerState* m_EmptySamplers[16];
		ID3D11UnorderedAccessView* m_EmptyUnorderedAccessViews[8];

		uint32_t m_CurrentCrc = UINT32_MAX;
		ID3D11InputLayout* m_InputLayout = nullptr;
		GfxRenderStateDX11 m_RenderState = {};
		GfxRenderStateCacheDX11 m_StateCache;
		GfxInputLayoutCacheDX11 m_LayoutCache;

		GfxBufferDX11* m_VertexBuffer = nullptr;
		GfxBufferDX11* m_IndexBuffer = nullptr;
		GfxBufferDX11* m_InstanceBuffer = nullptr;
		uint32_t m_InstanceOffset = 0;

		uint32_t m_ViewCount = 1;
		uint32_t m_DepthBias = 0;
		float m_SlopeDepthBias = 0;
		Topology m_Topology = (Topology)-1;

		friend class GfxRenderStateCacheDX11;
		friend class GfxInputLayoutCacheDX11;
	};
}