#include "SkyRendererEditor.h"

#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Editor\Assets\Processors\ReflectionGenerator.h"
#include "Editor\Panels\Scene\SceneArea.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void SkyRendererEditor::OnEnable()
	{
		m_MaterialProperty = m_SerializedObject->FindProperty("m_Material");
		m_ReflectionTextureProperty = m_SerializedObject->FindProperty("m_ReflectionTexture");
	}

	void SkyRendererEditor::OnDrawInspector()
	{
		ObjectEditor::OnDrawInspector();
		if (ImGui::Button("Bake"))
		{
			// Rewrite into .hdr with 16 bit
			Material* material = static_cast<Material*>(m_MaterialProperty.GetObjectPtr().Get());
			if (material != nullptr)
			{
				Texture* baseMap = material->GetTexture(TO_HASH("_BaseMap"));
				if (baseMap != nullptr)
				{
					m_ReflectionTextureProperty.SetObjectPtr(ReflectionGenerator::GenerateReflectionTexture(static_cast<TextureCube*>(baseMap)));
					m_SerializedObject->ApplyModifiedProperties();
					SceneArea::RequestRedrawAll();
				}
			}
		}
	}
}
