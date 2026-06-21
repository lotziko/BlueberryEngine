#include "HBAORendererDX12.h"

#include "..\DX12\GfxDeviceDX12.h"
#include "..\DX12\GfxTextureDX12.h"

#include <hbao\GFSDK_SSAO.h>

namespace Blueberry
{
	bool HBAORendererDX12::InitializeImpl()
	{
		m_GfxDevice = static_cast<GfxDeviceDX12*>(GfxDevice::GetInstance());
		GfxDescriptorRingHeapDX12& srvHeap = m_GfxDevice->GetCbvSrvDsvRingHeap();
		GfxDescriptorHeapDX12& rtvHeap = m_GfxDevice->GetRtvHeap();

		m_Device = m_GfxDevice->GetDevice();
		m_CommandQueue = m_GfxDevice->GetCommandQueue();
		m_CommandList = m_GfxDevice->GetCommandList();

		GFSDK_SSAO_CustomHeap customHeap;
		customHeap.new_ = ::operator new;
		customHeap.delete_ = ::operator delete;

		GFSDK_SSAO_DescriptorHeaps_D3D12 descriptorHeaps;
		descriptorHeaps.CBV_SRV_UAV.pDescHeap = srvHeap.GetHeap();
		descriptorHeaps.CBV_SRV_UAV.BaseIndex = m_SrvHeapIndex = srvHeap.AllocatePersistent(GFSDK_SSAO_NUM_DESCRIPTORS_CBV_SRV_UAV_HEAP_D3D12).GetIndex();
		descriptorHeaps.RTV.pDescHeap = rtvHeap.GetHeap();
		descriptorHeaps.RTV.BaseIndex = m_RtvHeapIndex = rtvHeap.Allocate(GFSDK_SSAO_NUM_DESCRIPTORS_RTV_HEAP_D3D12);

		GFSDK_SSAO_Status status;
		status = GFSDK_SSAO_CreateContext_D3D12(m_Device, 1, descriptorHeaps, &m_AOContext, &customHeap);
		if (status != GFSDK_SSAO_OK)
		{
			return false;
		}
		return true;
	}

	void HBAORendererDX12::ShutdownImpl()
	{
		GfxDescriptorHeapDX12& srvHeap = m_GfxDevice->GetCbvSrvDsvHeap();
		GfxDescriptorHeapDX12& rtvHeap = m_GfxDevice->GetRtvHeap();

		srvHeap.Free(m_SrvHeapIndex, GFSDK_SSAO_NUM_DESCRIPTORS_CBV_SRV_UAV_HEAP_D3D12);
		rtvHeap.Free(GFSDK_SSAO_NUM_DESCRIPTORS_RTV_HEAP_D3D12);

		m_AOContext->Release();
	}

	void HBAORendererDX12::DrawImpl(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* outputColor)
	{
		if (viewport != m_Viewport)
		{
			if (m_Viewport.width > 0)
			{
				m_GfxDevice->WaitForGPU();
				m_GfxDevice->Reset();
			}
			m_Viewport = viewport;
		}

		GfxTextureDX12* depthStencilTexture = static_cast<GfxTextureDX12*>(depthStencil);
		GfxTextureDX12* colorOutputTexture = static_cast<GfxTextureDX12*>(outputColor);

		GFSDK_SSAO_InputData_D3D12 input;
		input.DepthData.DepthTextureType = GFSDK_SSAO_HARDWARE_DEPTHS;
		input.DepthData.FullResDepthTextureSRV.pResource = depthStencilTexture->GetResource();
		input.DepthData.FullResDepthTextureSRV.GpuHandle = reinterpret_cast<UINT64>(depthStencilTexture->GetHandle());
		input.DepthData.ProjectionMatrix.Data = GFSDK_SSAO_Float4x4((const GFSDK_SSAO_FLOAT*)&projection);
		input.DepthData.ProjectionMatrix.Layout = GFSDK_SSAO_ROW_MAJOR_ORDER;
		input.DepthData.MetersToViewSpaceUnits = 1.0f;
		input.DepthData.Viewport.Enable = true;
		input.DepthData.Viewport.TopLeftX = viewport.x;
		input.DepthData.Viewport.TopLeftY = viewport.y;
		input.DepthData.Viewport.Width = viewport.width;
		input.DepthData.Viewport.Height = viewport.height;

		GFSDK_SSAO_Parameters params;
		params.Radius = 2.f;
		params.Bias = 0.1f;
		params.PowerExponent = 1.f;
		params.Blur.Enable = true;
		params.Blur.Radius = GFSDK_SSAO_BLUR_RADIUS_4;
		params.Blur.Sharpness = 16.f;

		GFSDK_SSAO_RenderTargetView_D3D12 rtv;
		rtv.pResource = colorOutputTexture->GetResource();
		rtv.CpuHandle = colorOutputTexture->GetRTV().GetCPU().ptr;

		GFSDK_SSAO_Output_D3D12 output;
		output.pRenderTargetView = &rtv;
		output.Blend.Mode = GFSDK_SSAO_OVERWRITE_RGB;

		D3D12_RESOURCE_BARRIER preBarriers[] = { CD3DX12_RESOURCE_BARRIER::Transition(depthStencilTexture->GetResource(), depthStencilTexture->GetState(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE), CD3DX12_RESOURCE_BARRIER::Transition(colorOutputTexture->GetResource(), colorOutputTexture->GetState(), D3D12_RESOURCE_STATE_RENDER_TARGET) };
		D3D12_RESOURCE_BARRIER postBarriers[] = { CD3DX12_RESOURCE_BARRIER::Transition(depthStencilTexture->GetResource(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, depthStencilTexture->GetState()), CD3DX12_RESOURCE_BARRIER::Transition(colorOutputTexture->GetResource(), D3D12_RESOURCE_STATE_RENDER_TARGET, colorOutputTexture->GetState()) };
		
		m_CommandList->ResourceBarrier(2, preBarriers);
		GFSDK_SSAO_Status status = m_AOContext->RenderAO(m_CommandQueue, m_CommandList, input, params, output);
		m_CommandList->ResourceBarrier(2, postBarriers);
		assert(status == GFSDK_SSAO_OK);
		m_GfxDevice->Reset();
	}
}