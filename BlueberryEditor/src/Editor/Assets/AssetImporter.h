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
		std::string GetFilePath();
		std::string GetMetaFilePath();
		const std::string& GetRelativeFilePath();
		const std::string& GetRelativeMetaFilePath();
		const std::vector<ObjectId>& GetImportedObjects();

		void Save();
		
		static AssetImporter* Create(const size_t& type, const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath);
		static AssetImporter* Load(const std::filesystem::path& relativePath, const std::filesystem::path& relativeMetaPath);

		static void BindProperties();
		
	protected:
		virtual void ImportData() = 0;
		void AddImportedObject(Object* object);

	private:
		Guid m_Guid;
		std::string m_RelativePath;
		std::string m_RelativeMetaPath;
		std::vector<ObjectId> m_ImportedObjects;
	};
}