#pragma once

#include "Editor\Inspector\ObjectInspector.h"

namespace Blueberry
{
	class LightInspector : public ObjectInspector
	{
	public:
		LightInspector();
		virtual ~LightInspector() = default;

		virtual Texture* GetIcon(Object* object) final;
		virtual void DrawScene(Object* object) override;

	private:
		void DrawCone(const float& radius, const float& height, const int& mask);
	};
}