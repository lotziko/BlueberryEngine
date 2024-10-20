#include "bbpch.h"
#include "HBAORendererDX11.h"

#include "hbao\GFSDK_SSAO.h"
#include "Blueberry\Graphics\RenderTexture.h"
#include "Concrete\DX11\GfxTextureDX11.h"

namespace Blueberry
{
	HBAORendererDX11::HBAORendererDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext) : m_Device(device), m_DeviceContext(deviceContext)
	{
	}

	bool HBAORendererDX11::Initialize()
	{
		GFSDK_SSAO_CustomHeap CustomHeap;
		CustomHeap.new_ = ::operator new;
		CustomHeap.delete_ = ::operator delete;

		GFSDK_SSAO_Status status;
		status = GFSDK_SSAO_CreateContext_D3D11(m_Device, &m_AOContext, &CustomHeap);
		if (status != GFSDK_SSAO_OK) // HBAO+ requires feature level 11_0 or above
		{
			return false;
		}
		return true;
	}

	void HBAORendererDX11::Draw(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* output)
	{
		//Matrix transformLH = Matrix::Identity;
		//transformLH._33 = -1;
		//Matrix viewLH = transformLH * view;
		//Matrix projectionLH = transformLH * projection;

		GFSDK_SSAO_InputData_D3D11 Input;
		Input.DepthData.DepthTextureType = GFSDK_SSAO_HARDWARE_DEPTHS;
		Input.DepthData.pFullResDepthTextureSRV = ((GfxTextureDX11*)depthStencil)->GetSRV();
		Input.DepthData.ProjectionMatrix.Data = GFSDK_SSAO_Float4x4((const GFSDK_SSAO_FLOAT*)&projection);
		Input.DepthData.ProjectionMatrix.Layout = GFSDK_SSAO_ROW_MAJOR_ORDER;
		Input.DepthData.MetersToViewSpaceUnits = 1.0f;
		Input.DepthData.Viewport.Enable = true;
		Input.DepthData.Viewport.TopLeftX = viewport.x;
		Input.DepthData.Viewport.TopLeftY = viewport.y;
		Input.DepthData.Viewport.Width = viewport.width;
		Input.DepthData.Viewport.Height = viewport.height;

		Input.NormalData.Enable = true;
		Input.NormalData.pFullResNormalTextureSRV = ((GfxTextureDX11*)normals)->GetSRV();
		Input.NormalData.WorldToViewMatrix.Data = GFSDK_SSAO_Float4x4((const GFSDK_SSAO_FLOAT*)&view);
		Input.NormalData.WorldToViewMatrix.Layout = GFSDK_SSAO_ROW_MAJOR_ORDER;
		Input.NormalData.DecodeScale = 2;
		Input.NormalData.DecodeBias = -1;

		GFSDK_SSAO_Parameters Params;
		Params.Radius = 2.f;
		Params.Bias = 0.1f;
		Params.PowerExponent = 2.f;
		Params.Blur.Enable = true;
		Params.Blur.Radius = GFSDK_SSAO_BLUR_RADIUS_4;
		Params.Blur.Sharpness = 16.f;

		GFSDK_SSAO_Output_D3D11 Output;
		Output.pRenderTargetView = ((GfxTextureDX11*)output)->GetRTV();
		Output.Blend.Mode = GFSDK_SSAO_OVERWRITE_RGB;

		GFSDK_SSAO_Status status = m_AOContext->RenderAO(m_DeviceContext, Input, Params, Output);
		assert(status == GFSDK_SSAO_OK);
	}
}
