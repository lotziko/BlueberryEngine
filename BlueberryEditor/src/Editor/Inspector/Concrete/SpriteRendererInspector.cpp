#include "bbpch.h"
#include "SpriteRendererInspector.h"

#include "Blueberry\Scene\Components\SpriteRenderer.h"

#include "imgui\imgui.h"
#include "Editor\Misc\ImGuiHelper.h"

namespace Blueberry
{
	OBJECT_INSPECTOR_DECLARATION(SpriteRendererInspector, SpriteRenderer)

	void SpriteRendererInspector::Draw(Object* object)
	{
		SpriteRenderer* spriteRenderer = static_cast<SpriteRenderer*>(object);

		Color color = spriteRenderer->GetColor();
		if (ImGui::ColorEdit("Color", color))
		{
			spriteRenderer->SetColor(color);
		}
	}
}