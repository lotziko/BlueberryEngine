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

		Mesh* GetMesh();
		void SetMesh(Mesh* mesh);

		Material* GetMaterial();
		void SetMaterial(Material* material);

		const AABB& GetBounds();

		static void BindProperties();

	private:
		ObjectPtr<Mesh> m_Mesh;
		ObjectPtr<Material> m_Material;
		AABB m_Bounds;
		size_t m_RecalculationFrame = 0;
	};
}