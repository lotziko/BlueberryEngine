#include "SpriteRendererInspector.h"

#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Scene\Components\SpriteRenderer.h"

#include "Editor\Misc\ImGuiHelper.h"

#include <imgui\imgui.h>

namespace Blueberry
{
	//void SpriteRendererInspector::Draw(Object* object)
	//{
	//	SpriteRenderer* spriteRenderer = static_cast<SpriteRenderer*>(object);

	//	Color color = spriteRenderer->GetColor();
	//	if (ImGui::ColorEdit("Color", color))
	//	{
	//		spriteRenderer->SetColor(color);
	//	}

	//	/*Texture2D* texture = spriteRenderer->GetTexture();
	//	if (ImGui::ObjectEdit("Texture", texture))
	//	{
	//		spriteRenderer->SetTexture(texture);
	//	}

	//	Material* material = spriteRenderer->GetMaterial();
	//	if (ImGui::ObjectEdit("Material", material))
	//	{
	//		spriteRenderer->SetMaterial(material);
	//	}*/
	//}
}