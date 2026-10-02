#include "GameViewRenderer.h"

#include "Blueberry\Core\Screen.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"
#include "Blueberry\Graphics\Concrete\DefaultRenderer.h"
#include "Blueberry\Scene\Scene.h"
#include "Blueberry\Scene\Components\Camera.h"
#include "Blueberry\Tools\CameraHelper.h"

namespace Blueberry
{
	GfxTexture* GameViewRenderer::s_RenderTarget = nullptr;

	void GameViewRenderer::Draw(Scene* scene)
	{
		Camera* mainCamera = nullptr;
		for (Camera* camera : scene->GetComponents<Camera>())
		{
			if (camera->GetEntity()->IsActiveInHierarchy())
			{
				mainCamera = camera;
				break;
			}
		}
		if (mainCamera != nullptr)
		{
			Camera::SetCurrent(mainCamera);
			RectangleFloat viewport = CameraHelper::CalculateViewport(mainCamera, Screen::GetGameViewport());

			if (s_RenderTarget == nullptr || static_cast<uint32_t>(viewport.width) != s_RenderTarget->GetWidth() || static_cast<uint32_t>(viewport.height) != s_RenderTarget->GetHeight())
			{
				if (s_RenderTarget != nullptr)
				{
					GfxTexturePool::Release(s_RenderTarget);
				}
				s_RenderTarget = GfxTexturePool::Get(static_cast<uint32_t>(viewport.width), static_cast<uint32_t>(viewport.height), 1, TextureUsageFlags::RenderTarget, 1, 1, TextureFormat::R8G8B8A8_UNorm);
				mainCamera->SetPixelSize(Vector2(viewport.width, viewport.height));
			}

			DefaultRenderer::Draw(scene, mainCamera, Rectangle(0l, 0l, static_cast<long>(viewport.width), static_cast<long>(viewport.height)), s_RenderTarget);
			GfxDevice::SetRenderTarget(nullptr);
			GfxDevice::SetViewport(static_cast<int>(viewport.x), static_cast<int>(viewport.y), static_cast<int>(viewport.width), static_cast<int>(viewport.height));
			GfxDevice::SetGlobalTexture(TO_HASH("_BlitTexture"), s_RenderTarget);
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetBlit()));
		}
	}
}