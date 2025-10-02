#pragma once

#include "Blueberry\Core\Base.h"
#include "Concrete\DX11\DX11.h"

namespace Blueberry
{
	class GfxDeviceDX11;
	class GfxVertexShaderDX11;
	class VertexLayout;

	class GfxInputLayoutCacheDX11
	{
	public:
		GfxInputLayoutCacheDX11() = default;

		void Shutdown();

		ID3D11InputLayout* GetLayout(GfxVertexShaderDX11* shader, VertexLayout* meshLayout);

	private:
		Dictionary<size_t, ID3D11InputLayout*> m_InputLayouts;
	};
}