#pragma once
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class AssetImporter : public Object
	{
		OBJECT_DECLARATION(AssetImporter)

	public:
		AssetImporter() = default;

		const Guid& GetGuid();
		const std::string& GetFilePath();
		const std::string& GetMetaFilePath();

		void Save();
		
		static AssetImporter* Create(const size_t& type, const std::string& path, const std::string& metaPath);
		static AssetImporter* Load(const std::string& path, const std::string& metaPath);

		static void BindProperties();
		
	protected:
		virtual void ImportData() = 0;

	private:
		Guid m_Guid;
		std::string m_Path;
		std::string m_MetaPath;
	};
}