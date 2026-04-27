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

		const String& GetName() const;
		void SetName(const String& name);

		const String& GetReplaceName() const;
		void SetReplaceName(const String& replaceName);

		uint32_t GetFirstFrame() const;
		void SetFirstFrame(uint32_t firstFrame);

		uint32_t GetLastFrame() const;
		void SetLastFrame(uint32_t lastFrame);

		bool IsLoop() const;
		void SetLoop(bool loop);

		String m_Name;
		String m_ReplaceName;
		uint32_t m_FirstFrame = 0;
		uint32_t m_LastFrame = 0;
		bool m_IsLoop = false;
	};

	class ModelImporter : public AssetImporter
	{
		OBJECT_DECLARATION(ModelImporter)

	public:
		ModelImporter() = default;

		List<ModelMaterialData>& GetMaterials();
		
		float GetScale() const;
		void SetScale(float scale);

		bool GetGenerateLightmapUV() const;
		void SetGenerateLightmapUV(bool generate);

		bool GetGeneratePhysicsShape() const;
		void SetGeneratePhysicsShape(bool generate);

	protected:
		virtual void ImportData() final;

	private:
		struct NodeData
		{
			Entity* entity;
			Transform* transform;
			String nodeName;
			String entityName;
		};

		void CreateHierarchy(Transform* parent, fbxsdk::FbxNode* node, List<Object*>& objects, Dictionary<fbxsdk::FbxNode*, NodeData>& nodeToData, float globalScale);
		void CreateMesh(fbxsdk::FbxNode* node, List<Object*>& objects, Dictionary<fbxsdk::FbxNode*, NodeData>& nodeToData, float globalScale);
		void CreateAnimationClips(fbxsdk::FbxScene* scene, List<Object*>& objects, float globalScale);
		
	private:
		List<ModelMaterialData> m_Materials;
		List<ModelAnimationClipData> m_AnimationClips;
		float m_Scale = 1.0f;
		bool m_GenerateLightmapUV = false;
		bool m_GeneratePhysicsShape = true;
	};
}