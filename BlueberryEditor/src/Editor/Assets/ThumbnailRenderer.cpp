#include "bbpch.h"
#include "ThumbnailRenderer.h"

#include "Blueberry\Graphics\RenderTexture.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\StandardMeshes.h"
#include "Blueberry\Graphics\DefaultMaterials.h"

#include "Editor\Preview\MaterialPreview.h"
#include "Editor\Preview\MeshPreview.h"

namespace Blueberry
{
	bool ThumbnailRenderer::CanDraw(const std::size_t& type)
	{
		return type == Texture2D::Type || type == Material::Type || type == Mesh::Type;
	}

	bool ThumbnailRenderer::Draw(unsigned char* output, const uint32_t& size, Object* asset)
	{
		if (s_ThumbnailRenderTarget == nullptr)
		{
			s_ThumbnailRenderTarget = RenderTexture::Create(size, size, 1, 1, TextureFormat::R8G8B8A8_UNorm, TextureDimension::Texture2D, WrapMode::Clamp, FilterMode::Linear, true);
		}

		if (asset->IsClassType(Texture2D::Type))
		{
			static std::size_t blitTextureId = TO_HASH("_BlitTexture");
			Texture2D* importedTexture = static_cast<Texture2D*>(asset);
			GfxDevice::SetRenderTarget(s_ThumbnailRenderTarget->Get());
			GfxDevice::SetViewport(0, 0, size, size);
			GfxDevice::SetGlobalTexture(blitTextureId, importedTexture->Get());
			GfxDevice::Draw(GfxDrawingOperation(StandardMeshes::GetFullscreen(), DefaultMaterials::GetBlit()));
			GfxDevice::Read(s_ThumbnailRenderTarget->Get(), output, Rectangle(0, 0, size, size));
			GfxDevice::SetRenderTarget(nullptr);
			return true;
		}
		else if (asset->IsClassType(Material::Type))
		{
			static MaterialPreview preview;
			preview.Draw(static_cast<Material*>(asset), s_ThumbnailRenderTarget);
			GfxDevice::Read(s_ThumbnailRenderTarget->Get(), output, Rectangle(0, 0, size, size));
			GfxDevice::SetRenderTarget(nullptr);
			return true;
		}
		else if (asset->IsClassType(Mesh::Type))
		{
			static MeshPreview preview;
			preview.Draw(static_cast<Mesh*>(asset), s_ThumbnailRenderTarget);
			GfxDevice::Read(s_ThumbnailRenderTarget->Get(), output, Rectangle(0, 0, size, size));
			GfxDevice::SetRenderTarget(nullptr);
			return true;
		}
		return false;
	}
}
