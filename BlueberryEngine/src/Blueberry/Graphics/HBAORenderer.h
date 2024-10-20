#pragma once

namespace Blueberry
{
	class GfxTexture;

	class HBAORenderer
	{
	public:
		virtual void Draw(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* output) = 0;
	};
}