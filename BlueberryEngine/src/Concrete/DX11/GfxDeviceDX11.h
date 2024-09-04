#pragma once

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	class GfxTextureDX11;
	class GfxConstantBufferDX11;
	class GfxVertexShaderDX11;
	class GfxGeometryShaderDX11;
	class GfxFragmentShaderDX11;

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

		virtual bool CreateVertexShaderImpl(void* vertexData, GfxVertexShader*& shader) final;
		virtual bool CreateGeometryShaderImpl(void* geometryData, GfxGeometryShader*& shader) final;
		virtual bool CreateFragmentShaderImpl(void* fragmentData, GfxFragmentShader*& shader) final;
		virtual bool CreateComputeShaderImpl(void* computeData, GfxComputeShader*& shader) final;
		virtual bool CreateVertexBufferImpl(const VertexLayout& layout, const UINT& vertexCount, GfxVertexBuffer*& buffer) final;
		virtual bool CreateIndexBufferImpl(const UINT& indexCount, GfxIndexBuffer*& buffer) final;
		virtual bool CreateConstantBufferImpl(const UINT& byteCount, GfxConstantBuffer*& buffer) final;
		virtual bool CreateComputeBufferImpl(const UINT& elementCount, const UINT& elementSize, GfxComputeBuffer*& buffer) final;
		virtual bool CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) const final;
		virtual bool CreateImGuiRendererImpl(ImGuiRenderer*& renderer) const final;

		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) const final;
		virtual void ReadImpl(GfxTexture* source, void* target, const Rectangle& area) const final;

		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture) final;
		virtual void SetGlobalConstantBufferImpl(const std::size_t& id, GfxConstantBuffer* buffer) final;
		virtual void SetGlobalTextureImpl(const std::size_t& id, GfxTexture* texture) final;
		virtual void DrawImpl(const GfxDrawingOperation& operation) final;

		virtual void DispatchImpl(GfxComputeShader*& shader, const UINT& threadGroupsX, const UINT& threadGroupsY, const UINT& threadGroupsZ) const final;

		virtual Matrix GetGPUMatrixImpl(const Matrix& viewProjection) const final;

	private:
		bool InitializeDirectX(HWND hwnd, int width, int height);

		void SetCullMode(const CullMode& mode);
		void SetBlendMode(const BlendMode& blendSrc, const BlendMode& blendDst);
		void SetZTestAndZWrite(const ZTest& zTest, const ZWrite& zWrite);
		void SetTopology(const Topology& topology);

		HWND m_Hwnd;

		ComPtr<ID3D11Device> m_Device;
		ComPtr<ID3D11DeviceContext> m_DeviceContext;
		ComPtr<IDXGISwapChain> m_SwapChain;
		ComPtr<ID3D11RenderTargetView> m_RenderTargetView;

		ComPtr<ID3D11RasterizerState> m_CullNoneRasterizerState;
		ComPtr<ID3D11RasterizerState> m_CullFrontRasterizerState;
		ComPtr<ID3D11RasterizerState> m_CullBackRasterizerState;

		std::unordered_map<std::size_t, ComPtr<ID3D11DepthStencilState>> m_DepthStencilStates;
		std::unordered_map<std::size_t, ComPtr<ID3D11BlendState>> m_BlendStates;

		GfxTextureDX11* m_BindedRenderTarget;
		GfxTextureDX11* m_BindedDepthStencil;
		std::unordered_map<std::size_t, GfxConstantBufferDX11*> m_BindedConstantBuffers;
		ID3D11Buffer* m_ConstantBuffers[8];
		std::unordered_map<std::size_t, GfxTextureDX11*> m_BindedTextures;
		ID3D11ShaderResourceView* m_ShaderResourceViews[16];
		ID3D11SamplerState* m_Samplers[16];

		CullMode m_CullMode = (CullMode)-1;
		std::size_t m_BlendKey = -1;
		std::size_t m_ZTestZWriteKey = -1;
		Topology m_Topology = (Topology)-1;
		
		ObjectId m_MaterialId = 0;
		GfxVertexShaderDX11* m_VertexShader = nullptr;
		GfxGeometryShaderDX11* m_GeometryShader = nullptr;
		GfxFragmentShaderDX11* m_FragmentShader = nullptr;
	};
}