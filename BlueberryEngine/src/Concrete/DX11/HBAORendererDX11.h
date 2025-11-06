#pragma once

#include "..\..\Blueberry\Graphics\HBAORenderer.h"
#include "Concrete\DX11\DX11.h"

class GFSDK_SSAO_Context_D3D11;

namespace Blueberry
{
	class HBAORendererDX11 : public HBAORenderer
	{
	protected:
		virtual bool InitializeImpl() final;
		virtual void ShutdownImpl() final;

		virtual void DrawImpl(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* output) final;

	private:
		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;
		GFSDK_SSAO_Context_D3D11* m_AOContext;
	};
}