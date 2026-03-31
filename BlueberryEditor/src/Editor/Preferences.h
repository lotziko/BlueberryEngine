#pragma once

namespace Blueberry
{
	class Preferences
	{
	public:
		static float* GetGizmoSnapping();

		static int GetGizmoOperation();
		static void SetGizmoOperation(int operation);

	private:
		static float s_GizmoSnapping[3];
		static int s_GizmoOperation;
	};
}