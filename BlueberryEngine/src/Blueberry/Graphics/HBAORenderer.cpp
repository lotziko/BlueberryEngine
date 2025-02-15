#include "bbpch.h"
#include "HBAORenderer.h"

#include "GraphicsAPI.h"
#include "Concrete\DX11\HBAORendererDX11.h"

namespace Blueberry
{
	bool HBAORenderer::Initialize()
	{
		switch (GraphicsAPI::GetAPI())
		{
		case GraphicsAPI::API::None:
			BB_ERROR("API doesn't exist.");
			return false;
		case GraphicsAPI::API::DX11:
			s_Instance = new HBAORendererDX11();
		}
		return s_Instance->InitializeImpl();
	}

	void HBAORenderer::Shutdown()
	{

	}

	void HBAORenderer::Draw(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* output)
	{
		s_Instance->DrawImpl(depthStencil, normals, view, projection, viewport, output);
	}
}