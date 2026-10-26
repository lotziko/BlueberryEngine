#pragma once

#include "Editor\Inspector\Concrete\AssetImporterEditor.h"

namespace Blueberry
{
	class GfxTexture;

	class ModelImporterEditor : public AssetImporterEditor
	{
	public:
		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

	private:
		SerializedProperty m_MaterialsProperty;
		SerializedProperty m_AnimationClipsProperty;
		SerializedProperty m_ScaleProperty;
		SerializedProperty m_GenerateLightmapUVProperty;
		SerializedProperty m_GeneratePhysicsShapeProperty;

		static GfxTexture* s_RenderTexture;
	};
}