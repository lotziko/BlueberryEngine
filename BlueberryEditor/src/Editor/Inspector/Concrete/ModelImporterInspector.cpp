#include "bbpch.h"
#include "ModelImporterInspector.h"

#include "Blueberry\Graphics\Material.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\ModelImporter.h"
#include "Editor\Misc\ImGuiHelper.h"

#include "imgui\imgui.h"

namespace Blueberry
{
	void ModelImporterInspector::Draw(Object* object)
	{
		ModelImporter* modelImporter = static_cast<ModelImporter*>(object);
		modelImporter->ImportDataIfNeeded();

		for (auto& materialPtr : modelImporter->GetMaterials())
		{
			ModelMaterialData* data = materialPtr.Get();
			Material* material = data->GetMaterial();
			if (ImGui::ObjectEdit(data->GetName().c_str(), (Object**)&material, Material::Type))
			{
				data->SetMaterial(material);
			}
		}

		if (ImGui::Button("Save"))
		{
			AssetDB::SetDirty(object);
			AssetDB::SaveAssets();
		}
	}
}
