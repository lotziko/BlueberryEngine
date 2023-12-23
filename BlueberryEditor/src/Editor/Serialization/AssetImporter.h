#pragma once
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Guid.h"
#include <filesystem>

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
		const std::vector<ObjectId>& GetImportedObjects();

		void Save();
		
		static AssetImporter* Create(const size_t& type, const std::filesystem::path& path, const std::filesystem::path& metaPath);
		static AssetImporter* Load(const std::filesystem::path& path, const std::filesystem::path& metaPath);

		static void BindProperties();
		
	protected:
		virtual void ImportData() = 0;
		void AddImportedObject(Object* object);

	private:
		Guid m_Guid;
		std::string m_Path;
		std::string m_MetaPath;
		std::vector<ObjectId> m_ImportedObjects;
	};
}