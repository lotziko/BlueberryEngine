#pragma once

#include "Editor\Inspector\Concrete\AssetImporterEditor.h"

namespace Blueberry
{
	class TextureImporterEditor : public AssetImporterEditor
	{
	public:
		virtual void OnEnable() override;
		virtual void OnDrawInspector() override;

	private:
		SerializedProperty m_GenerateMipmapsProperty;
		SerializedProperty m_IsSRGBProperty;
		SerializedProperty m_WrapModeProperty;
		SerializedProperty m_FilterModeProperty;
		SerializedProperty m_TextureShapeProperty;
		SerializedProperty m_TextureTypeProperty;
		SerializedProperty m_TextureFormatProperty;
		SerializedProperty m_TextureCubeTypeProperty;
		SerializedProperty m_TextureCubeIBLTypeProperty;
	};
}