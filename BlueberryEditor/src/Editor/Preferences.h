#pragma once

namespace Blueberry
{
	class Preferences
	{
	public:
		static float* GetGizmoSnapping();

		static int GetGizmoOperation();
		static void SetGizmoOperation(int operation);

		static bool IsGizmoLocalSpace();
		static void SetGizmoLocalSpace(bool isLocalSpace);

	private:
		static float s_GizmoSnapping[3];
		static int s_GizmoOperation;
		static bool s_IsGizmoLocalSpace;
	};
}