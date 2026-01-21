#pragma once

#include "Editor\Panels\EditorWindow.h"

namespace Blueberry
{
	class AnimationGraphWindow : public EditorWindow
	{
		OBJECT_DECLARATION(AnimationGraphWindow)

	public:
		AnimationGraphWindow() = default;
		virtual ~AnimationGraphWindow() = default;

		static void Open();

		virtual void OnDrawUI() final;
	};
}