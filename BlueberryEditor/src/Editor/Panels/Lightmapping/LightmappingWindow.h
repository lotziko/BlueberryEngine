#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class LightmappingWindow : public EditorWindow
	{
		OBJECT_DECLARATION(LightmappingWindow)

	public:
		LightmappingWindow() = default;
		virtual ~LightmappingWindow() = default;

		static void Open();

		virtual void OnDrawUI() final;
	};
}