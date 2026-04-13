#pragma once

#include "Renderer.h"

namespace Blueberry
{
	class Mesh;
	class Material;
	class GfxBuffer;

	class BB_API SkinnedMeshRenderer : public Renderer
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

		const bool HasRoot();
		Transform* GetRoot();
		void SetRoot(Transform* root);

		void SetBones(const List<Transform*>& bones);

		virtual const AABB& GetBounds() final;
		virtual const Matrix& GetLocalToWorldMatrix() final;

		const List<Matrix>& GetWorldMatrices();
		const List<Matrix>& GetSkinningMatrices();

	private:
		bool CalculateSkinning();
		bool IsSkinningBufferValid();

		void UpdateBounds();
		void InvalidateBounds();
		void UpdateBoneDatas();

	private:
		struct BoneData
		{
			ObjectPtr<Transform> transform;
			int32_t parentIndex;
		};

		ObjectPtr<Mesh> m_Mesh;
		List<ObjectPtr<Material>> m_Materials;
		ObjectPtr<Transform> m_Root;
		List<ObjectPtr<Transform>> m_Bones;
		AABB m_PreviousBounds;
		AABB m_Bounds = AABB(Vector3::Zero, Vector3::Zero);
		size_t m_UpdateCount = 0;
		size_t m_LastVisibleFrame = 0;
		List<Matrix> m_LocalMatrices;
		List<Matrix> m_WorldMatrices;
		List<Matrix> m_SkinningMatrices;
		List<BoneData> m_BoneDatas;
		bool m_CullingDirty = true;

		GfxBuffer* m_SkinningVertexBuffer = nullptr;
		ObjectId m_SkinningMeshId = INVALID_ID;
		uint32_t m_SkinningMeshUpdateCount = 0;

		friend class Skinning;
	};
}