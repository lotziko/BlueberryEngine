#pragma once

namespace Blueberry
{
	class Preferences
	{
	public:
		static float* GetGizmoSnapping();

		static const int& GetGizmoOperation();
		static void SetGizmoOperation(const int& operation);

	private:
		static float s_GizmoSnapping[3];
		static int s_GizmoOperation;
	};
}