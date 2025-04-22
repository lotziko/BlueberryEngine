#pragma once

namespace Blueberry
{
	class GfxDeviceDX11;
	class GfxVertexShaderDX11;
	class VertexLayout;

	class GfxInputLayoutCacheDX11
	{
	public:
		GfxInputLayoutCacheDX11() = default;

		ID3D11InputLayout* GetLayout(GfxVertexShaderDX11* shader, VertexLayout* meshLayout);

	private:
		Dictionary<size_t, ID3D11InputLayout*> m_InputLayouts;
	};
}