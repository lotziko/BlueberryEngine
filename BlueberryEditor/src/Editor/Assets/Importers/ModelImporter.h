#pragma once
#include "Editor\Assets\AssetImporter.h"

#include "Blueberry\Core\ObjectPtr.h"

namespace fbxsdk
{
	class FbxNode;
	class FbxScene;
}

namespace Blueberry
{
	class Material;
	class Transform;
	class Entity;

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

	class ModelAnimationClipData : public Data
	{
		DATA_DECLARATION(ModelAnimationClipData)

	public:
		ModelAnimationClipData() = default;
		virtual ~ModelAnimationClipData() = default;

		const String& GetName();
		void SetName(const String& name);

		const String& GetReplaceName();
		void SetReplaceName(const String& replaceName);

		const uint32_t& GetFirstFrame();
		void SetFirstFrame(const uint32_t& firstFrame);

		const uint32_t& GetLastFrame();
		void SetLastFrame(const uint32_t& lastFrame);

		String m_Name;
		String m_ReplaceName;
		uint32_t m_FirstFrame;
		uint32_t m_LastFrame;
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
		struct NodeData
		{
			Entity* entity;
			Transform* transform;
			String nodeName;
			String entityName;
		};

		void CreateHierarchy(Transform* parent, fbxsdk::FbxNode* node, List<Object*>& objects, Dictionary<fbxsdk::FbxNode*, NodeData>& nodeToData, const float& globalScale);
		void CreateMesh(fbxsdk::FbxNode* node, List<Object*>& objects, Dictionary<fbxsdk::FbxNode*, NodeData>& nodeToData, const float& globalScale);
		void CreateAnimationClips(fbxsdk::FbxScene* scene, List<Object*>& objects, const float& globalScale);
		
	private:
		List<ModelMaterialData> m_Materials;
		List<ModelAnimationClipData> m_AnimationClips;
		float m_Scale = 1.0f;
		bool m_GenerateLightmapUV = false;
		bool m_GeneratePhysicsShape = true;
	};
}