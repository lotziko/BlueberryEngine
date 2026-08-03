#pragma once

#include "..\..\Blueberry\Graphics\HBAORenderer.h"
#include "GfxDescriptorHeapDX12.h"
#include "Concrete\DX12\DX12.h"

class GFSDK_SSAO_Context_D3D12;

namespace Blueberry
{
	class GfxDeviceDX12;

	class HBAORendererDX12 : public HBAORenderer
	{
	protected:
		virtual bool InitializeImpl() final;
		virtual void ShutdownImpl() final;

		virtual void DrawImpl(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* outputColor) final;

	private:
		ID3D12Device* m_Device;
		GfxDeviceDX12* m_GfxDevice;
		ID3D12CommandQueue* m_CommandQueue;
		ID3D12GraphicsCommandList* m_CommandList;
		GFSDK_SSAO_Context_D3D12* m_AOContext;
		GfxHandleDX12 m_SrvHandle;
		GfxHandleDX12 m_RtvHandle;
		Rectangle m_Viewport;
	};
}