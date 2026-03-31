#pragma once

#include "Blueberry\Core\Base.h"
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Structs.h"
#include "Blueberry\Graphics\Structs.h"

namespace Blueberry
{
	class GfxTexture;

	class BB_API Texture : public Object
	{
		OBJECT_DECLARATION(Texture)

	public:
		Texture() = default;
		virtual ~Texture() = default;

		uint32_t GetWidth() const;
		uint32_t GetHeight() const;
		GfxTexture* Get();
		void* GetHandle();

		WrapMode GetWrapMode() const;
		void SetWrapMode(WrapMode wrapMode);

		FilterMode GetFilterMode() const;
		void SetFilterMode(FilterMode filterMode);

		bool IsReadable() const;
		void SetReadable(bool readable);

	protected:
		void IncrementUpdateCount();

	protected:
		GfxTexture* m_Texture = nullptr;
		uint32_t m_Width = 0;
		uint32_t m_Height = 0;
		uint32_t m_MipCount = 0;
		TextureFormat m_Format = TextureFormat::R8G8B8A8_UNorm;
		TextureDimension m_Dimension = TextureDimension::Texture2D;
		WrapMode m_WrapMode = WrapMode::Clamp;
		FilterMode m_FilterMode = FilterMode::Bilinear;
		bool m_IsReadable = true;
		ByteData m_RawData = {};

		uint32_t m_UpdateCount = 0;
		HashSet<ObjectId> m_Dependencies;

		friend class Material;
		friend struct GfxDrawingOperation;
		friend struct GfxRenderState;
	};
}