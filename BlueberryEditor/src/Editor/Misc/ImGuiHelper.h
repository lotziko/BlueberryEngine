#pragma once

namespace ImGui
{
	bool DragVector3(const std::string& label, Blueberry::Vector3& vector);
	bool ColorEdit(const std::string& label, Blueberry::Color& color);

	void ApplyEditorDarkTheme();
	void LoadDefaultEditorFonts();
}