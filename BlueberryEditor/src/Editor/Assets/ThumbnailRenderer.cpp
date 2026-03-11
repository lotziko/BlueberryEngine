#include "ThumbnailRenderer.h"

#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\GfxTexturePool.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"

#include "Editor\Preview\MaterialPreview.h"
#include "Editor\Preview\MeshPreview.h"

namespace Blueberry
{
	GfxTexture* ThumbnailRenderer::s_ThumbnailRenderTarget = nullptr;

	bool ThumbnailRenderer::CanDraw(const TypeId& type)
	{
		return type == Texture2D::Type || type == Material::Type || type == Mesh::Type;
	}

	bool ThumbnailRenderer::Draw(unsigned char* output, const uint32_t& size, Object* asset)
	{
		if (asset->GetState() != ObjectState::Default)
		{
			return false;
		}

		if (s_ThumbnailRenderTarget == nullptr)
		{
			s_ThumbnailRenderTarget = GfxTexturePool::Get(size, size, 1, TextureUsageFlags::RenderTarget | TextureUsageFlags::CPUReadable, 1, 1, TextureFormat::R8G8B8A8_UNorm, TextureDimension::Texture2D, WrapMode::Clamp, FilterMode::Bilinear);
		}

		if (asset->IsClassType(Texture2D::Type))
		{
			static size_t blitTextureId = TO_HASH("_BlitTexture");
			Texture2D* importedTexture = static_cast<Texture2D*>(asset);
			GfxDevice::SetRenderTarget(s_ThumbnailRenderTarget);
			GfxDevice::SetViewport(0, 0, size, size);
			GfxDevice::SetGlobalTexture(blitTextureId, importedTexture->Get());
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetBlit()));
			GfxDevice::SetRenderTarget(nullptr);
			s_ThumbnailRenderTarget->GetData(output, Rectangle(0, 0, size, size));
			return true;
		}
		else if (asset->IsClassType(Material::Type))
		{
			static MaterialPreview preview;
			preview.Draw(static_cast<Material*>(asset), s_ThumbnailRenderTarget);
			GfxDevice::SetRenderTarget(nullptr);
			s_ThumbnailRenderTarget->GetData(output, Rectangle(0, 0, size, size));
			return true;
		}
		else if (asset->IsClassType(Mesh::Type))
		{
			static MeshPreview preview;
			preview.Draw(static_cast<Mesh*>(asset), s_ThumbnailRenderTarget);
			GfxDevice::SetRenderTarget(nullptr);
			s_ThumbnailRenderTarget->GetData(output, Rectangle(0, 0, size, size));
			return true;
		}
		return false;
	}
}
