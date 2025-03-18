#pragma once

namespace Blueberry
{
	class RenderTexture;
	struct CameraData;

	class OpenXRRenderer
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		static bool Initialize();
		static void Shutdown();
		static bool IsActive();

		static void BeginFrame();
		static void FillCameraData(CameraData& cameraData);
		static void SubmitColorRenderTarget(RenderTexture* renderTarget);
		static void EndFrame();

	protected:
		virtual bool InitializeImpl() = 0;
		virtual void ShutdownImpl() = 0;
		virtual bool IsActiveImpl() = 0;

		virtual void BeginFrameImpl() = 0;
		virtual void FillCameraDataImpl(CameraData& cameraData) = 0;
		virtual void SubmitColorRenderTargetImpl(RenderTexture* renderTarget) = 0;
		virtual void EndFrameImpl() = 0;

	private:
		static inline OpenXRRenderer* s_Instance = nullptr;
	};
}