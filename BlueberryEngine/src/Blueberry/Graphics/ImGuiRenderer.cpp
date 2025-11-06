#include "Blueberry\Graphics\ImGuiRenderer.h"

#include "GraphicsAPI.h"
#include "..\..\Concrete\DX11\ImGuiRendererDX11.h"

namespace Blueberry
{
	bool ImGuiRenderer::Initialize()
	{
		switch (GraphicsAPI::GetAPI())
		{
		case GraphicsAPI::API::None:
			BB_ERROR("API doesn't exist.");
			return false;
		case GraphicsAPI::API::DX11:
			s_Instance = new ImGuiRendererDX11();
		}
		return s_Instance->InitializeImpl();
	}

	void ImGuiRenderer::Shutdown()
	{
		s_Instance->ShutdownImpl();
	}

	void ImGuiRenderer::Begin()
	{
		s_Instance->BeginImpl();
	}

	void ImGuiRenderer::End()
	{
		s_Instance->EndImpl();
	}
}
