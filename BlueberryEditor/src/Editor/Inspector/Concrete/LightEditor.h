#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class LightEditor : public ObjectEditor
	{
	public:
		LightEditor();
		virtual ~LightEditor() = default;

		virtual Texture* GetIcon(Object* object) final;
		virtual void OnDrawSceneSelected() override;

	private:
		void DrawCone(const float& radius, const float& height, const int& mask);
	};
}