#pragma once

namespace ImGui
{
	bool DragVector3(const std::string& label, Vector3& vector);

	void ApplyEditorDarkTheme();
	void LoadDefaultEditorFonts();
}