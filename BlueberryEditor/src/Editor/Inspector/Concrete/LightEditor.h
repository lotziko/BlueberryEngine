#pragma once

#include "Editor\Inspector\ObjectEditor.h"

namespace Blueberry
{
	class LightEditor : public ObjectEditor
	{
	public:
		LightEditor();
		virtual ~LightEditor() = default;

		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

		virtual Texture* GetIcon(Object* object) final;
		virtual void OnDrawSceneSelected() override;

	private:
		void DrawCone(const float& radius, const float& height, const int& mask);
	
	private:
		SerializedProperty m_TypeProperty;
		SerializedProperty m_ColorProperty;
		SerializedProperty m_IntensityProperty;
		SerializedProperty m_RangeProperty;
		SerializedProperty m_OuterSpotAngleProperty;
		SerializedProperty m_InnerSpotAngleProperty;
		SerializedProperty m_IsCastingShadowsProperty;
		SerializedProperty m_IsCastingFogProperty;
		SerializedProperty m_IsCachedProperty;
		SerializedProperty m_CookieProperty;
	};
}