#include "HBAORenderer.h"

#include "Blueberry\Graphics\GraphicsAPI.h"

#include "..\..\Concrete\DX11\HBAORendererDX11.h"
#include "..\..\Concrete\DX12\HBAORendererDX12.h"

namespace Blueberry
{
	HBAORenderer* HBAORenderer::s_Instance = nullptr;

	bool HBAORenderer::Initialize()
	{
		switch (GraphicsAPI::GetAPI())
		{
		case GraphicsAPI::API::None:
			BB_ERROR("API doesn't exist.");
			return false;
		case GraphicsAPI::API::DX11:
			s_Instance = new HBAORendererDX11();
			break;
		case GraphicsAPI::API::DX12:
			s_Instance = new HBAORendererDX12();
			break;
		}
		return s_Instance->InitializeImpl();
	}

	void HBAORenderer::Shutdown()
	{
		s_Instance->ShutdownImpl();
	}

	void HBAORenderer::Draw(GfxTexture* depthStencil, GfxTexture* normals, const Matrix& view, const Matrix& projection, const Rectangle& viewport, GfxTexture* outputColor)
	{
		s_Instance->DrawImpl(depthStencil, normals, view, projection, viewport, outputColor);
	}
}