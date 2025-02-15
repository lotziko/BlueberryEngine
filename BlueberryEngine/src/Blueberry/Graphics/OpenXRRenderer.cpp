#include "bbpch.h"
#include "OpenXRRenderer.h"

#include "GraphicsAPI.h"

#include "Concrete\DX11\OpenXRRendererDX11.h"

namespace Blueberry
{
	bool OpenXRRenderer::Initialize()
	{
		switch (GraphicsAPI::GetAPI())
		{
		case GraphicsAPI::API::None:
			BB_ERROR("API doesn't exist.");
			return false;
		case GraphicsAPI::API::DX11:
			s_Instance = new OpenXRRendererDX11();
		}
		return s_Instance->InitializeImpl();
	}

	void OpenXRRenderer::Shutdown()
	{
		s_Instance->ShutdownImpl();
	}

	bool OpenXRRenderer::IsActive()
	{
		if (s_Instance == nullptr)
		{
			return false;
		}

		return s_Instance->IsActiveImpl();
	}

	void OpenXRRenderer::BeginFrame()
	{
		if (s_Instance == nullptr)
		{
			return;
		}

		s_Instance->BeginFrameImpl();
	}

	void OpenXRRenderer::FillCameraData(CameraData& cameraData)
	{
		if (s_Instance == nullptr)
		{
			return;
		}

		s_Instance->FillCameraDataImpl(cameraData);
	}

	void OpenXRRenderer::SubmitColorRenderTarget(RenderTexture* renderTarget)
	{
		if (s_Instance == nullptr)
		{
			return;
		}

		s_Instance->SubmitColorRenderTargetImpl(renderTarget);
	}

	void OpenXRRenderer::EndFrame()
	{
		if (s_Instance == nullptr)
		{
			return;
		}

		s_Instance->EndFrameImpl();
	}
}
