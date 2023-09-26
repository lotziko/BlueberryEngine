#include "bbpch.h"
#include "EditorMaterials.h"

#include "Blueberry\Core\GlobalServices.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"

namespace Blueberry
{
	Ref<Material> EditorMaterials::s_EditorGridMaterial = nullptr;
	bool EditorMaterials::s_EditorGridMaterialTriedToLoad = false;

	Material* EditorMaterials::GetEditorGridMaterial()
	{
		if (s_EditorGridMaterialTriedToLoad == false)
		{
			Ref<Shader> shaderRef;
			g_AssetManager->Load<Shader>("assets/EditorGrid", shaderRef);
			s_EditorGridMaterial = Material::Create(shaderRef);
			s_EditorGridMaterialTriedToLoad = true;
		}

		return s_EditorGridMaterial.get();
	}
}