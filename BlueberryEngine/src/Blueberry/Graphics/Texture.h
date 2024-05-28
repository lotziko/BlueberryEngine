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
		ByteData m_RawData = {};

		friend class Material;
		friend class GfxDrawingOperation;
		friend class Renderer2D;
	};
}