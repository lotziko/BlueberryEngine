#include "SkyRendererInspector.h"

#include "Blueberry\Graphics\TextureCube.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Components\SkyRenderer.h"
#include "Editor\Assets\Processors\ReflectionGenerator.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	void SkyRendererInspector::Draw(Object* object)
	{
		ObjectInspector::Draw(object);
		SkyRenderer* skyRenderer = static_cast<SkyRenderer*>(object);
		if (ImGui::Button("Bake"))
		{
			// Rewrite into .hdr with 16 bit
			Material* material = skyRenderer->GetMaterial();
			if (material != nullptr)
			{
				Texture* baseMap = material->GetTexture(TO_HASH("_BaseMap"));
				if (baseMap != nullptr)
				{
					skyRenderer->SetReflectionTexture(ReflectionGenerator::GenerateReflectionTexture(static_cast<TextureCube*>(baseMap)));
				}
			}
		}
	}
}
