#include "ModelImporterInspector.h"

#include "Blueberry\Graphics\Material.h"
#include "Editor\Assets\AssetDB.h"
#include "Editor\Assets\Importers\ModelImporter.h"
#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void ModelImporterInspector::Draw(Object* object)
	{
		ModelImporter* modelImporter = static_cast<ModelImporter*>(object);
		modelImporter->ImportDataIfNeeded();

		for (auto& materialData : modelImporter->GetMaterials())
		{
			Material* material = materialData.GetMaterial();
			if (ImGui::ObjectEdit(materialData.GetName().c_str(), (Object**)&material, Material::Type))
			{
				materialData.SetMaterial(material);
			}
		}

		float scale = modelImporter->GetScale();
		if (ImGui::FloatEdit("Scale", &scale))
		{
			modelImporter->SetScale(scale);
		}

		bool generateLightmapUV = modelImporter->GetGenerateLightmapUV();
		if (ImGui::BoolEdit("Generate Lightmap UV", &generateLightmapUV))
		{
			modelImporter->SetGenerateLightmapUV(generateLightmapUV);
		}

		bool generatePhysicsShape = modelImporter->GetGeneratePhysicsShape();
		if (ImGui::BoolEdit("Generate Physics Shape", &generatePhysicsShape))
		{
			modelImporter->SetGeneratePhysicsShape(generatePhysicsShape);
		}

		if (ImGui::Button("Save"))
		{
			AssetDB::SetDirty(object);
			AssetDB::SaveAssets();
		}
	}
}
