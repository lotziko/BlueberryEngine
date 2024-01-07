#pragma once

namespace Blueberry
{
	class Material;

	class EditorMaterials
	{
	public:
		static Material* GetEditorColorMaterial();

	private:
		static Material* s_EditorColorMaterial;
		static bool s_EditorColorMaterialTriedToLoad;
	};
}