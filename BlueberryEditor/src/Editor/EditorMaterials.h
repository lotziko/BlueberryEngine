#pragma once

namespace Blueberry
{
	class Material;

	class EditorMaterials
	{
	public:
		static Material* GetEditorGridMaterial();

	private:
		static Ref<Material> s_EditorGridMaterial;
		static bool s_EditorGridMaterialTriedToLoad;
	};
}