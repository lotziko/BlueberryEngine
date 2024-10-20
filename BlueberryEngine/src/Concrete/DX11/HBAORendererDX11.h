#pragma once

#include "Blueberry\Graphics\HBAORenderer.h"

class GFSDK_SSAO_Context_D3D11;

namespace Blueberry
{
	class HBAORendererDX11 : public HBAORenderer
	{
	public:
		HBAORendererDX11(ID3D11Device* device, ID3D11DeviceContext* deviceContext);

		bool Initialize();

		virtual void Draw(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* output) final;

	private:
		ID3D11Device* m_Device;
		ID3D11DeviceContext* m_DeviceContext;
		GFSDK_SSAO_Context_D3D11* m_AOContext;
	};
}