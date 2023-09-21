#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Graphics\GfxTexture.h"

namespace Blueberry
{
	class Texture : public Object
	{
		OBJECT_DECLARATION(Texture)

	public:
		Texture() = default;
		virtual ~Texture() = default;

		inline UINT GetWidth() const
		{
			return m_Texture->GetWidth();
		}

		inline UINT GetHeight() const
		{
			return m_Texture->GetHeight();
		}

		inline void* GetHandle()
		{
			return m_Texture->GetHandle();
		}

	protected:
		Ref<GfxTexture> m_Texture;

		friend struct GfxDrawingOperation;
	};
}