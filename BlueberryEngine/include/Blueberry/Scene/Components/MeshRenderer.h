#pragma once

#include "Renderer.h"
#include "Transform.h"
#include "Blueberry\Graphics\CullableInterface.h"

namespace Blueberry
{
	class Mesh;
	class Material;
	class GfxBottomLevelAccelerationStructure;

	class BB_API MeshRenderer : public Renderer, public TransformDependencyInterface, public CullableInterface
	{
		OBJECT_DECLARATION(MeshRenderer)

	public:
		MeshRenderer() = default;
		virtual ~MeshRenderer() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;

		virtual void OnTransformInvalidate() final;
		virtual void OnPreCull() final;

		Mesh* GetMesh();
		void SetMesh(Mesh* mesh);

		Material* GetMaterial(uint32_t index = 0) const;
		void SetMaterial(Material* material);

		const List<ObjectPtr<Material>>& GetMaterials() const;
		void SetMaterials(const List<Material*> materials);

		uint32_t GetMaterialCount() const;

		virtual const AABB& GetBounds() final;
		virtual const Matrix& GetLocalToWorldMatrix() final;

		const bool& IsBakeable();

		uint32_t GetLightmapChartOffset() const;
		void SetLightmapChartOffset(uint32_t offset);

		GfxBottomLevelAccelerationStructure* GetAccelerationStructure();

	private:
		void UpdateBounds();
		void InvalidateBounds();

	private:
		ObjectPtr<Mesh> m_Mesh;
		List<ObjectPtr<Material>> m_Materials;
		AABB m_Bounds = AABB(Vector3::Zero, Vector3::Zero);
		bool m_IsBakeable = false;
		bool m_BoundsDirty = true;
		uint32_t m_LightmapChartOffset = 0;

		GfxBottomLevelAccelerationStructure* m_AccelerationStructure = nullptr;
		uint32_t m_MeshUpdateCount = 0;
	};
}