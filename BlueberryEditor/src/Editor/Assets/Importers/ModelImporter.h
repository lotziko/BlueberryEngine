#pragma once
#include "Editor\Assets\AssetImporter.h"

#include "Blueberry\Core\ObjectPtr.h"

namespace fbxsdk
{
	class FbxNode;
}

namespace Blueberry
{
	class Material;
	class Transform;

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

	private:
		std::string m_Name;
		ObjectPtr<Material> m_Material;
	};

	class ModelImporter : public AssetImporter
	{
		OBJECT_DECLARATION(ModelImporter)

	public:
		ModelImporter() = default;

		DataList<ModelMaterialData>& GetMaterials();
		
		const float& GetScale();
		void SetScale(const float& scale);

	protected:
		virtual void ImportData() override;

	private:
		void CreateMeshEntity(Transform* parent, fbxsdk::FbxNode* node, List<Object*>& objects);

	private:
		DataList<ModelMaterialData> m_Materials;
		float m_Scale = 1.0f;
	};
}