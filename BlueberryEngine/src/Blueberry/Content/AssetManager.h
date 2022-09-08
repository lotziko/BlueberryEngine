#pragma once

#include "Blueberry\Content\AssetImporter.h"

namespace Blueberry
{
	class AssetManager
	{
	public:
		AssetManager();
		~AssetManager() = default;

		template<class ContentType>
		bool Load(const std::string& path, Ref<ContentType>& content);

	private:
		void Register(AssetImporter* importer);

	private:
		std::map<std::string, Ref<Object>> m_LoadedContent;
		std::map<std::size_t, AssetImporter*> m_Importers;
	};

	template<class ContentType>
	inline bool AssetManager::Load(const std::string& path, Ref<ContentType>& content)
	{
		Ref<Object> ref;
		static_assert(std::is_base_of<Object, ContentType>::value, "Type is not base.");
		std::size_t type = ContentType::Type;

		if (!m_LoadedContent.count(path))
		{
			if (m_Importers.count(type) == 0)
			{
				return false;
			}

			AssetImporter* importer = m_Importers.find(type)->second;
			ref = importer->Import(path);
			m_LoadedContent.insert({ path, ref });
		}
		else
		{
			ref = m_LoadedContent[path];
		}

		content = std::dynamic_pointer_cast<ContentType>(ref);
		return true;
	}
}