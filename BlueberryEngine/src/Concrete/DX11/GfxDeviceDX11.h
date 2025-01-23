#pragma once

#include "Blueberry\Graphics\GfxDevice.h"

namespace Blueberry
{
	class GfxTextureDX11;
	class GfxConstantBufferDX11;
	class GfxStructuredBufferDX11;
	class GfxVertexShaderDX11;
	class GfxGeometryShaderDX11;
	class GfxFragmentShaderDX11;
	class GfxVertexBufferDX11;
	class GfxIndexBufferDX11;

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
		virtual void SetScissorRectImpl(int x, int y, int width, int height) final;
		virtual void ResizeBackbufferImpl(int width, int height) final;

		virtual const uint32_t& GetViewCountImpl() final;
		virtual void SetViewCountImpl(const uint32_t& count) final;

		virtual bool CreateVertexShaderImpl(void* vertexData, GfxVertexShader*& shader) final;
		virtual bool CreateGeometryShaderImpl(void* geometryData, GfxGeometryShader*& shader) final;
		virtual bool CreateFragmentShaderImpl(void* fragmentData, GfxFragmentShader*& shader) final;
		virtual bool CreateComputeShaderImpl(void* computeData, GfxComputeShader*& shader) final;
		virtual bool CreateVertexBufferImpl(const VertexLayout& layout, const uint32_t& vertexCount, GfxVertexBuffer*& buffer) final;
		virtual bool CreateIndexBufferImpl(const uint32_t& indexCount, GfxIndexBuffer*& buffer) final;
		virtual bool CreateConstantBufferImpl(const uint32_t& byteCount, GfxConstantBuffer*& buffer) final;
		virtual bool CreateStructuredBufferImpl(const uint32_t& elementCount, const uint32_t& elementSize, GfxStructuredBuffer*& buffer) final;
		virtual bool CreateComputeBufferImpl(const uint32_t& elementCount, const uint32_t& elementSize, GfxComputeBuffer*& buffer) final;
		virtual bool CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) const final;
		virtual bool CreateImGuiRendererImpl(ImGuiRenderer*& renderer) const final;
		virtual bool CreateHBAORendererImpl(HBAORenderer*& renderer) const final;

		virtual void CopyImpl(GfxTexture* source, GfxTexture* target) const final;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) const final;
		virtual void ReadImpl(GfxTexture* source, void* target, const Rectangle& area) const final;

		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture) final;
		virtual void SetGlobalConstantBufferImpl(const size_t& id, GfxConstantBuffer* buffer) final;
		virtual void SetGlobalStructuredBufferImpl(const size_t& id, GfxStructuredBuffer* buffer) final;
		virtual void SetGlobalTextureImpl(const size_t& id, GfxTexture* texture) final;
		virtual void DrawImpl(const GfxDrawingOperation& operation) final;

		virtual void DispatchImpl(GfxComputeShader*& shader, const uint32_t& threadGroupsX, const uint32_t& threadGroupsY, const uint32_t& threadGroupsZ) const final;

		virtual Matrix GetGPUMatrixImpl(const Matrix& viewProjection) const final;

	private:
		bool InitializeDirectX(HWND hwnd, int width, int height);

		void SetCullMode(const CullMode& mode);
		void SetBlendMode(const BlendMode& blendSrcColor, const BlendMode& blendSrcAlpha, const BlendMode& blendDstColor, const BlendMode& blendDstAlpha);
		void SetZTestAndZWrite(const ZTest& zTest, const ZWrite& zWrite);
		void SetTopology(const Topology& topology);

		const uint32_t& GetCRC();

		HWND m_Hwnd;

		ComPtr<ID3D11Device> m_Device;
		ComPtr<ID3D11DeviceContext> m_DeviceContext;
		ComPtr<IDXGISwapChain> m_SwapChain;
		ComPtr<ID3D11RenderTargetView> m_RenderTargetView;

		ComPtr<ID3D11RasterizerState> m_CullNoneRasterizerState;
		ComPtr<ID3D11RasterizerState> m_CullFrontRasterizerState;
		ComPtr<ID3D11RasterizerState> m_CullBackRasterizerState;

		std::unordered_map<size_t, ComPtr<ID3D11DepthStencilState>> m_DepthStencilStates;
		std::unordered_map<size_t, ComPtr<ID3D11BlendState>> m_BlendStates;

		GfxTextureDX11* m_BindedRenderTarget;
		GfxTextureDX11* m_BindedDepthStencil;
		std::unordered_map<size_t, GfxConstantBufferDX11*> m_BindedConstantBuffers;
		ID3D11Buffer* m_ConstantBuffers[8];
		std::unordered_map<size_t, GfxStructuredBufferDX11*> m_BindedStructuredBuffers;
		std::unordered_map<size_t, GfxTextureDX11*> m_BindedTextures;
		ID3D11ShaderResourceView* m_ShaderResourceViews[16];
		ID3D11SamplerState* m_Samplers[16];

		ObjectId m_MaterialId = 0;
		uint32_t m_MaterialCrc = 0;
		uint32_t m_GlobalCrc = 0;
		uint32_t m_CurrentCrc = 0;
		CullMode m_CullMode = (CullMode)-1;
		size_t m_BlendKey = -1;
		size_t m_ZTestZWriteKey = -1;
		Topology m_Topology = (Topology)-1;
		
		GfxVertexShaderDX11* m_VertexShader = nullptr;
		GfxGeometryShaderDX11* m_GeometryShader = nullptr;
		GfxFragmentShaderDX11* m_FragmentShader = nullptr;
		GfxRenderState* m_RenderState = nullptr;

		GfxVertexBufferDX11* m_VertexBuffer = nullptr;
		GfxIndexBufferDX11* m_IndexBuffer = nullptr;
		GfxVertexBufferDX11* m_InstanceBuffer = nullptr;
		uint32_t m_InstanceOffset = 0;

		uint32_t m_ViewCount = 1;
	};
}