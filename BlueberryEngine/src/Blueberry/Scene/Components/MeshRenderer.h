#pragma once

#include "Renderer.h"

namespace Blueberry
{
	class Mesh;
	class Material;

	class MeshRenderer : public Renderer
	{
		OBJECT_DECLARATION(MeshRenderer)

	public:
		MeshRenderer() = default;
		virtual ~MeshRenderer() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;
		
		void Update();

		Mesh* GetMesh();
		void SetMesh(Mesh* mesh);

		Material* GetMaterial(const uint32_t& index = 0);
		void SetMaterial(Material* material);

		void SetMaterials(const List<Material*> materials);

		virtual const AABB& GetBounds() final;

		static void BindProperties();

	private:
		ObjectPtr<Mesh> m_Mesh;
		List<ObjectPtr<Material>> m_Materials;
		AABB m_Bounds;
		AABB m_TreeBounds;
		size_t m_RecalculationFrame = 0;
	};
}