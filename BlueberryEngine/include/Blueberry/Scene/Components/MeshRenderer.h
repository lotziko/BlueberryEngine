#pragma once

#include "Renderer.h"

namespace Blueberry
{
	class Mesh;
	class Material;

	class BB_API MeshRenderer : public Renderer
	{
		OBJECT_DECLARATION(MeshRenderer)

	public:
		MeshRenderer() = default;
		virtual ~MeshRenderer() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;
		
		void OnPreCull();

		Mesh* GetMesh();
		void SetMesh(Mesh* mesh);

		Material* GetMaterial(const uint32_t& index = 0);
		void SetMaterial(Material* material);

		void SetMaterials(const List<Material*> materials);

		virtual const AABB& GetBounds() final;

		const uint32_t& GetLightmapChartOffset();
		void SetLightmapChartOffset(const uint32_t& offset);

	private:
		void UpdateBounds();

	private:
		ObjectPtr<Mesh> m_Mesh;
		List<ObjectPtr<Material>> m_Materials;
		AABB m_PreviousBounds;
		AABB m_Bounds;
		size_t m_RecalculationFrame = 0;
		bool m_CullingDirty = true;
		uint32_t m_LightmapChartOffset = 0;
	};
}