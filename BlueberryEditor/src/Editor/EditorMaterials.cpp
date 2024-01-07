#include "bbpch.h"
#include "EditorMaterials.h"

#include "Editor\Assets\EditorAssets.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"

namespace Blueberry
{
	Material* EditorMaterials::s_EditorColorMaterial = nullptr;
	bool EditorMaterials::s_EditorColorMaterialTriedToLoad = false;

	Material* EditorMaterials::GetEditorColorMaterial()
	{
		if (s_EditorColorMaterialTriedToLoad == false)
		{
			Shader* shaderRef = (Shader*)EditorAssets::Load("assets/Color.shader");
			s_EditorColorMaterial = Material::Create(shaderRef);
			s_EditorColorMaterialTriedToLoad = true;
		}

		return s_EditorColorMaterial;
	}
}