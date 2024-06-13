#pragma once
#include "Editor\Assets\AssetImporter.h"

#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\DataPtr.h"

namespace Blueberry
{
	class Material;

	class ModelMaterialData : public Data
	{
		DATA_DECLARATION(ModelMaterialData)

	public:
		ModelMaterialData() = default;
		virtual ~ModelMaterialData() = default;

		const std::string& GetName();
		void SetName(const std::string& name);

		Material* GetMaterial();
		void SetMaterial(Material* material);

		static void BindProperties();

	private:
		std::string m_Name;
		ObjectPtr<Material> m_Material;
	};

	class ModelImporter : public AssetImporter
	{
		OBJECT_DECLARATION(ModelImporter)

	public:
		ModelImporter() = default;

		const std::vector<DataPtr<ModelMaterialData>>& GetMaterials();

		static void BindProperties();

	protected:
		virtual void ImportData() override;
		virtual std::string GetIconPath() final;

	private:
		std::vector<DataPtr<ModelMaterialData>> m_Materials;
	};
}