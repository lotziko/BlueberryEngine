#pragma once

#include "Renderer.h"

namespace Blueberry
{
	class Mesh;
	class Material;
	class GfxBuffer;

	class SkinnedMeshRenderer : public Renderer
	{
		OBJECT_DECLARATION(SkinnedMeshRenderer)

	public:
		SkinnedMeshRenderer() = default;
		virtual ~SkinnedMeshRenderer() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;

		void OnPreCull();

		Mesh* GetMesh();
		void SetMesh(Mesh* mesh);

		Material* GetMaterial(const uint32_t& index = 0);
		void SetMaterial(Material* material);

		void SetMaterials(const List<Material*> materials);

		uint32_t GetMaterialCount();

		Transform* GetRoot();
		void SetRoot(Transform* root);

		virtual const AABB& GetBounds() final;

		const List<Matrix>& GetWorldMatrices();
		const List<Matrix>& GetSkinningMatrices();

	private:
		bool CalculateSkinning();
		bool IsSkinningBufferValid();

		void UpdateBounds();
		void InvalidateBounds();
		void GatherBones(Transform* parent = nullptr);

	private:
		struct BoneData
		{
			ObjectPtr<Transform> transform;
			int32_t parentIndex;
		};

		ObjectPtr<Mesh> m_Mesh;
		List<ObjectPtr<Material>> m_Materials;
		ObjectPtr<Transform> m_Root;
		AABB m_PreviousBounds;
		AABB m_Bounds = AABB(Vector3::Zero, Vector3::Zero);
		size_t m_UpdateCount = 0;
		size_t m_LastVisibleFrame = 0;
		List<Matrix> m_LocalMatrices;
		List<Matrix> m_WorldMatrices;
		List<Matrix> m_SkinningMatrices;
		List<BoneData> m_Bones;
		bool m_CullingDirty = true;

		GfxBuffer* m_SkinningVertexBuffer;
		ObjectId m_SkinningMeshId;
		uint32_t m_SkinningMeshUpdateCount;

		friend class Skinning;
	};
}