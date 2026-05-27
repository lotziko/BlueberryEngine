#pragma once

#include "Blueberry\Graphics\GfxDevice.h"
#include "Concrete\Windows\ComPtr.h"
#include "Concrete\DX12\DX12.h"

#include "GfxDescriptorHeapDX12.h"
#include "GfxUploadBufferDX12.h"
#include "GfxReadbackBufferDX12.h"
#include "GfxRenderStateCacheDX12.h"
#include "GfxComputeRenderStateCacheDX12.h"

namespace Blueberry
{
	class GfxTextureDX12;
	class GfxBufferDX12;

	class GfxDeviceDX12 final : public GfxDevice
	{
	public:
		GfxDeviceDX12() = default;
		virtual ~GfxDeviceDX12() = default;

	protected:
		virtual bool InitializeImpl(int width, int height, void* data) final;

		virtual void ClearColorImpl(const Color& color) final;
		virtual void ClearDepthImpl(float depth) final;
		virtual void WaitForFrameImpl() final;
		virtual void SwapBuffersImpl() final;

		virtual void SetViewportImpl(int x, int y, int width, int height) final;
		virtual void SetScissorRectImpl(int x, int y, int width, int height) final;
		virtual void ResizeBackbufferImpl(int width, int height) final;

		virtual uint32_t GetViewCountImpl() final;
		virtual void SetViewCountImpl(uint32_t count) final;
		virtual void SetDepthBiasImpl(uint32_t bias, float slopeBias) final;

		virtual bool CreateVertexShaderImpl(const ByteData& vertexData, GfxVertexShader*& shader) final;
		virtual bool CreateGeometryShaderImpl(const ByteData& geometryData, GfxGeometryShader*& shader) final;
		virtual bool CreateFragmentShaderImpl(const ByteData& fragmentData, GfxFragmentShader*& shader) final;
		virtual bool CreateComputeShaderImpl(const ByteData& computeData, GfxComputeShader*& shader) final;
		virtual bool CreateBufferImpl(const BufferProperties& properties, GfxBuffer*& buffer) final;
		virtual bool CreateTextureImpl(const TextureProperties& properties, GfxTexture*& texture) final;

		virtual void CopyImpl(GfxTexture* source, GfxTexture* target) final;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Rectangle& area) final;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, const Vector2Int& offset, const Rectangle& area) final;
		virtual void CopyImpl(GfxTexture* source, GfxTexture* target, uint32_t sourceSlice, uint32_t targetSlice, uint32_t mipLevel) final;

		virtual void SetRenderTargetImpl(GfxTexture* renderTexture, GfxTexture* depthStencilTexture, uint32_t arraySlice, uint32_t mipLevel) final;
		virtual void SetGlobalBufferImpl(size_t id, GfxBuffer* buffer) final;
		virtual void SetGlobalTextureImpl(size_t id, GfxTexture* texture) final;
		virtual void DrawImpl(const GfxDrawingOperation& operation) final;

		virtual void DispatchImpl(GfxComputeShader* shader, uint32_t threadGroupsX, uint32_t threadGroupsY, uint32_t threadGroupsZ) final;

		virtual Matrix GetGPUMatrixImpl(const Matrix& matrix) const final;

	public:
		ID3D12Device* GetDevice();
		ID3D12CommandQueue* GetCommandQueue();
		ID3D12GraphicsCommandList* GetCommandList();
		ID3D12RootSignature* GetGraphicsRootSignature();
		ID3D12RootSignature* GetComputeRootSignature();
		HWND GetHwnd();

		GfxDescriptorHeapDX12& GetCbvSrvDsvHeap();
		GfxDescriptorHeapDX12& GetRtvHeap();
		GfxDescriptorHeapDX12& GetDsvHeap();
		GfxDescriptorRingHeapDX12& GetCbvSrvDsvRingHeap();
		GfxUploadBufferDX12& GetUploadBuffer();
		GfxReadbackBufferDX12& GetReadbackBuffer();
		uint64_t GetGeneration();

		void WaitForGPU();
		void Reset();

		void Release(ComPtr<ID3D12Resource>& resource);

		void CreateCBV(D3D12_CONSTANT_BUFFER_VIEW_DESC* cbvDesc, GfxHandleDX12* handle);
		void CreateSRV(ID3D12Resource* resource, D3D12_SHADER_RESOURCE_VIEW_DESC* srvDesc, GfxHandleDX12* handle);
		void CreateUAV(ID3D12Resource* resource, D3D12_UNORDERED_ACCESS_VIEW_DESC* uavDesc, GfxHandleDX12* handle);
		void CreateRTV(ID3D12Resource* resource, D3D12_RENDER_TARGET_VIEW_DESC* rtvDesc, GfxHandleDX12* handle);
		void CreateDSV(ID3D12Resource* resource, D3D12_DEPTH_STENCIL_VIEW_DESC* dsvDesc, GfxHandleDX12* handle);
		void CreateSampler(D3D12_SAMPLER_DESC* samplerDesc, GfxHandleDX12* handle);

	private:
		bool InitializeDirectX(HWND hwnd, int width, int height);

		uint32_t GetSampler(WrapMode wrapMode, FilterMode filterMode);
		uint32_t GetSamplersOffset(const uint8_t* indexes, uint32_t size);
		void ResizeBackbufferIfNeeded();
		
		static const uint32_t BUFFER_COUNT = 2;

		struct FrameContext
		{
			ComPtr<ID3D12CommandAllocator> commandAllocator;
			UINT64 fenceValue;
		};

		struct BackbufferData
		{
			ComPtr<ID3D12Resource> resource;
			GfxHandleDX12 renderTargetView;
			D3D12_RESOURCE_STATES state;
		};

		struct SamplerOffsetData
		{
			uint8_t data[16];
			uint32_t offset;
		};

		HWND m_Hwnd;

		ComPtr<ID3D12Device> m_Device;
		ComPtr<IDXGISwapChain3> m_SwapChain;
		BackbufferData m_Backbuffers[BUFFER_COUNT];
		ComPtr<ID3D12CommandQueue> m_CommandQueue;
		ComPtr<ID3D12GraphicsCommandList> m_CommandList;
		ComPtr<ID3D12Fence> m_Fence;
		ComPtr<ID3D12RootSignature> m_GraphicsRootSignature;
		ComPtr<ID3D12RootSignature> m_ComputeRootSignature;
		FrameContext m_FrameContexts[BUFFER_COUNT];
		FrameContext* m_CurrentContext = nullptr;
		HANDLE m_FenceEvent;
		HANDLE m_FrameLatencyWaitHandle;
		UINT64 m_FenceLastSignaledValue;
		UINT m_FrameIndex;
		UINT m_BackbufferIndex;

		GfxDescriptorHeapDX12 m_CbvSrvUavHeap;
		GfxDescriptorHeapDX12 m_RtvHeap;
		GfxDescriptorHeapDX12 m_DsvHeap;
		GfxDescriptorHeapDX12 m_SamplerHeap;
		GfxDescriptorRingHeapDX12 m_CbvSrvUavRingHeap;
		GfxDescriptorRingHeapDX12 m_SamplerRingHeap;
		GfxUploadBufferDX12 m_UploadBuffer;
		GfxReadbackBufferDX12 m_ReadbackBuffer;
		List<std::pair<size_t, GfxHandleDX12>> m_Samplers;
		List<SamplerOffsetData> m_SamplerHeapOffsets;

		GfxTextureDX12* m_BindedRenderTarget;
		GfxTextureDX12* m_BindedDepthStencil;
		List<std::pair<size_t, uint32_t>> m_BindedBuffers;
		List<std::pair<size_t, uint32_t>> m_BindedTextures;
		GfxTargetInfoDX12 m_TargetInfo;
		List<ComPtr<ID3D12Resource>> m_ReleasedResources;

		GfxRenderStateCacheDX12 m_StateCache;
		GfxComputeRenderStateCacheDX12 m_ComputeStateCache;

		GfxBufferDX12* m_VertexBuffer = nullptr;
		GfxBufferDX12* m_IndexBuffer = nullptr;
		GfxBufferDX12* m_InstanceBuffer = nullptr;
		uint32_t m_InstanceOffset = 0;

		uint32_t m_ViewCount = 1;
		uint32_t m_DepthBias = 0;
		float m_SlopeDepthBias = 0;
		Topology m_Topology = (Topology)-1;
		D3D12_VIEWPORT m_Viewport;
		D3D12_RECT m_ScissorRect;
		Vector2Int m_BackbufferResizeRequest;

		friend class GfxDescriptorHeapDX12;
		friend class GfxRenderStateCacheDX12;
		friend class GfxComputeRenderStateCacheDX12;
	};
}