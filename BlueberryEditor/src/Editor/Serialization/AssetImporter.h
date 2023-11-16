#pragma once
#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\Guid.h"
#include "yaml-cpp\yaml.h"

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
		
		static Ref<AssetImporter> Create(const size_t& type, const std::string& path, const std::string& metaPath);
		static Ref<AssetImporter> Load(const std::string& path, const std::string& metaPath);
		
	protected:
		virtual void SerializeMeta(YAML::Emitter& out) = 0;
		virtual void DeserializeMeta(YAML::Node& in) = 0;
		virtual void ImportData() = 0;

	private:
		Guid m_Guid;
		std::string m_Type;
		std::string m_Path;
		std::string m_MetaPath;
	};
}