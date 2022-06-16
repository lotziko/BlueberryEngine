#pragma once

#include "Blueberry\Graphics\GraphicsDevice.h"
#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class ContentManager
	{
	public:
		ContentManager() = default;
		~ContentManager() = default;

		template<class ContentType>
		void Load(const std::string& path, Ref<ContentType>& content);

	private:
		std::map<std::string, Ref<Object>> m_LoadedContent;
	};
	template<class ContentType>
	inline void ContentManager::Load(const std::string& path, Ref<ContentType>& content)
	{
		static_assert(std::is_base_of<Object, ContentType>::value, "Type is not base.");
		static Ref<ContentType> ref;

		if (!m_LoadedContent.count(path))
		{
			if (ContentType::Type == Texture::Type)
			{
				g_GraphicsDevice->CreateTexture(path, ref);
				m_LoadedContent.insert({ path, ref });
			}

			content = ref;
		}
		else
		{
			content = std::dynamic_pointer_cast<ContentType>(m_LoadedContent[path]);
		}
	}
}