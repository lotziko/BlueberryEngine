#pragma once

#include "Blueberry\Core\Base.h"
#include "..\..\Blueberry\Graphics\OpenXRRenderer.h"

namespace Blueberry
{
	class GfxTexture;

	class OpenXRRendererDX11 : public OpenXRRenderer
	{
	protected:
		virtual bool InitializeImpl() final;
		virtual void ShutdownImpl() final;
		virtual bool IsActiveImpl() final;

		virtual void BeginFrameImpl() final;
		virtual void FillCameraDataImpl(CameraData& cameraData) final;
		virtual void SubmitColorRenderTargetImpl(GfxTexture* renderTarget) final;
		virtual void EndFrameImpl() final;

	private:
		Matrix m_MultiviewViewMatrix[2];
		Matrix m_MultiviewProjectionMatrix[2];
		Rectangle m_MultiviewViewport;
		GfxTexture* m_SubmittedColorRenderTarget;
	};
}