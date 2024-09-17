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

		Mesh* GetMesh();
		void SetMesh(Mesh* mesh);

		Material* GetMaterial(const UINT& index = 0);
		void SetMaterial(Material* material);

		void SetMaterials(const std::vector<Material*> materials);

		const AABB& GetBounds();

		static void BindProperties();

	private:
		ObjectPtr<Mesh> m_Mesh;
		//ObjectPtr<Material> m_Material;
		std::vector<ObjectPtr<Material>> m_Materials;
		AABB m_Bounds;
		size_t m_RecalculationFrame = 0;
	};
}