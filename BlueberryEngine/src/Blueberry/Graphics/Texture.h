#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Structs.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\Structs.h"

namespace Blueberry
{
	class Texture : public Object
	{
		OBJECT_DECLARATION(Texture)

	public:
		virtual ~Texture() = default;

		const UINT& GetWidth();
		const UINT& GetHeight();
		void* GetHandle();

		static void BindProperties();

	protected:
		GfxTexture* m_Texture = nullptr;
		UINT m_Width = 0;
		UINT m_Height = 0;
		UINT m_MipCount = 0;
		TextureFormat m_Format = TextureFormat::R8G8B8A8_UNorm;
		WrapMode m_WrapMode = WrapMode::Clamp;
		FilterMode m_FilterMode = FilterMode::Linear;
		ByteData m_RawData = {};

		friend class Material;
		friend struct GfxDrawingOperation;
		friend class Renderer2D;
	};
}