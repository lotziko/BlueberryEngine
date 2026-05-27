#include "HBAORendererDX11.h"

#include "..\DX11\GfxDeviceDX11.h"
#include "..\DX11\GfxTextureDX11.h"

#include <hbao\GFSDK_SSAO.h>

namespace Blueberry
{
	bool HBAORendererDX11::InitializeImpl()
	{
		GfxDeviceDX11* gfxDevice = static_cast<GfxDeviceDX11*>(GfxDevice::GetInstance());

		m_Device = gfxDevice->GetDevice();
		m_DeviceContext = gfxDevice->GetDeviceContext();

		GFSDK_SSAO_CustomHeap customHeap;
		customHeap.new_ = ::operator new;
		customHeap.delete_ = ::operator delete;

		GFSDK_SSAO_Status status;
		status = GFSDK_SSAO_CreateContext_D3D11(m_Device, &m_AOContext, &customHeap);
		if (status != GFSDK_SSAO_OK)
		{
			return false;
		}
		return true;
	}

	void HBAORendererDX11::ShutdownImpl()
	{
		m_AOContext->Release();
	}

	void HBAORendererDX11::DrawImpl(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* colorOutput)
	{
		GFSDK_SSAO_InputData_D3D11 input;
		input.DepthData.DepthTextureType = GFSDK_SSAO_HARDWARE_DEPTHS;
		input.DepthData.pFullResDepthTextureSRV = (static_cast<GfxTextureDX11*>(depthStencil))->GetSRV();
		input.DepthData.ProjectionMatrix.Data = GFSDK_SSAO_Float4x4((const GFSDK_SSAO_FLOAT*)&projection);
		input.DepthData.ProjectionMatrix.Layout = GFSDK_SSAO_ROW_MAJOR_ORDER;
		input.DepthData.MetersToViewSpaceUnits = 1.0f;
		input.DepthData.Viewport.Enable = true;
		input.DepthData.Viewport.TopLeftX = viewport.x;
		input.DepthData.Viewport.TopLeftY = viewport.y;
		input.DepthData.Viewport.Width = viewport.width;
		input.DepthData.Viewport.Height = viewport.height;

		//input.NormalData.Enable = true;
		/*input.NormalData.pFullResNormalTextureSRV = (static_cast<GfxTextureDX11*>(normals))->GetSRV();
		Input.NormalData.WorldToViewMatrix.Data = GFSDK_SSAO_Float4x4((const GFSDK_SSAO_FLOAT*)&view);
		input.NormalData.WorldToViewMatrix.Layout = GFSDK_SSAO_ROW_MAJOR_ORDER;
		input.NormalData.DecodeScale = 2;
		input.NormalData.DecodeBias = -1;*/

		GFSDK_SSAO_Parameters params;
		params.Radius = 2.f;
		params.Bias = 0.1f;
		params.PowerExponent = 1.f;
		params.Blur.Enable = true;
		params.Blur.Radius = GFSDK_SSAO_BLUR_RADIUS_4;
		params.Blur.Sharpness = 16.f;

		GFSDK_SSAO_Output_D3D11 output;
		output.pRenderTargetView = (static_cast<GfxTextureDX11*>(colorOutput))->GetRTV();
		output.Blend.Mode = GFSDK_SSAO_OVERWRITE_RGB;

		GFSDK_SSAO_Status status = m_AOContext->RenderAO(m_DeviceContext, input, params, output);
		assert(status == GFSDK_SSAO_OK);
	}
}
