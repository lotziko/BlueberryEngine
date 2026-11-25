#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class DebugWindow : public EditorWindow
	{
		OBJECT_DECLARATION(DebugWindow)

	public:
		DebugWindow() = default;
		virtual ~DebugWindow() = default;

		static void Open();

		virtual void OnDrawUI() final;
	};
}