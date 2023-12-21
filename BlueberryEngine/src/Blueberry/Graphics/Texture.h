#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\GfxTexture.h"
#include "Blueberry\Graphics\Structs.h"

namespace Blueberry
{
	class Texture : public Object
	{
		OBJECT_DECLARATION(Texture)

	public:
		Texture() = default;
		virtual ~Texture() = default;

		const UINT& GetWidth();
		const UINT& GetHeight();
		void* GetHandle();

		static void BindProperties();

	protected:
		GfxTexture* m_Texture;
		UINT m_Width;
		UINT m_Height;
		ByteData m_RawData;

		friend class Material;
	};
}