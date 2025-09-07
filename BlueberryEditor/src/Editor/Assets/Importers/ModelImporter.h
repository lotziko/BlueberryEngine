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

		const String& GetName();
		void SetName(const String& name);

		Material* GetMaterial();
		void SetMaterial(Material* material);

	private:
		String m_Name;
		ObjectPtr<Material> m_Material;
	};

	class ModelImporter : public AssetImporter
	{
		OBJECT_DECLARATION(ModelImporter)

	public:
		ModelImporter() = default;

		List<ModelMaterialData>& GetMaterials();
		
		const float& GetScale();
		void SetScale(const float& scale);

		const bool& GetGenerateLightmapUV();
		void SetGenerateLightmapUV(const bool& generate);

		const bool& GetGeneratePhysicsShape();
		void SetGeneratePhysicsShape(const bool& generate);

	protected:
		virtual void ImportData() override;

	private:
		void CreateMeshEntity(Transform* parent, fbxsdk::FbxNode* node, List<Object*>& objects);
		std::string GetPhysicsShapePath(const size_t& fileId);

	private:
		List<ModelMaterialData> m_Materials;
		float m_Scale = 1.0f;
		bool m_GenerateLightmapUV = false;
		bool m_GeneratePhysicsShape = true;
	};
}