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

		static void BindProperties();

	protected:
		Ref<GfxTexture> m_Texture;
		BYTE* m_RawData;
		size_t m_RawDataSize;

		friend class Material;
	};
}