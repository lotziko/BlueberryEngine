#pragma once

#include "Blueberry\Core\Guid.h"

namespace Blueberry
{
	class AssetImporter;

	class ImporterInfoCache
	{
	private:
		struct ImporterInfo
		{
			void Read(std::istream& is)
			{
				is.read((char*)&mainObject, sizeof(FileId));
				size_t objectCount;
				is.read((char*)&objectCount, sizeof(size_t));

				for (size_t i = 0; i < objectCount; ++i)
				{
					FileId objectFileId;
					is.read((char*)&objectFileId, sizeof(FileId));

					size_t type;
					is.read((char*)&type, sizeof(size_t));

					size_t nameSize;
					is.read((char*)&nameSize, sizeof(size_t));
					std::string name(nameSize, ' ');
					is.read(name.data(), nameSize);

					objects.emplace_back(std::tuple { objectFileId, type, name });
				}
			}

			void Write(std::ostream& os) const
			{
				os.write((char*)&mainObject, sizeof(FileId));

				size_t objectCount = objects.size();
				os.write((char*)&objectCount, sizeof(size_t));

				for (auto& tuple : objects)
				{
					FileId objectFileId = std::get<0>(tuple);
					os.write((char*)&objectFileId, sizeof(FileId));

					size_t type = std::get<1>(tuple);
					os.write((char*)&type, sizeof(size_t));

					std::string name = std::get<2>(tuple);
					size_t nameSize = name.size();
					os.write((char*)&nameSize, sizeof(size_t));
					os.write(name.data(), nameSize);
				}
			}

			FileId mainObject;
			std::vector<std::tuple<FileId, size_t, std::string>> objects;
		};

	public:
		static void Load();
		static void Save();

		static bool Has(AssetImporter* importer);
		static bool Get(AssetImporter* importer);
		static void Set(AssetImporter* importer);

	private:
		static std::unordered_map<Guid, ImporterInfo> m_ImporterInfoCache;
	};
}