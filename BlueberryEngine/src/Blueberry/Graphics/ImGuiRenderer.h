#pragma once

namespace Blueberry
{
	class ImGuiRenderer
	{
	public:
		virtual ~ImGuiRenderer() = default;

		virtual void Begin() = 0;
		virtual void End() = 0;
	};
}