#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxTexture;

	class HBAORenderer
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		static bool Initialize();
		static void Shutdown();

		static void Draw(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* output);

	protected:
		virtual bool InitializeImpl() = 0;
		virtual void DrawImpl(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* output) = 0;
	
	private:
		static inline HBAORenderer* s_Instance = nullptr;
	};
}