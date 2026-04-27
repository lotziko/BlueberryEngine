#include "Preferences.h"

namespace Blueberry
{
	float Preferences::s_GizmoSnapping[3] = { 0.1f, 5.0f, 1.0f };
	int Preferences::s_GizmoOperation = 7;

	float* Preferences::GetGizmoSnapping()
	{
		return s_GizmoSnapping;
	}

	int Preferences::GetGizmoOperation()
	{
		return s_GizmoOperation;
	}

	void Preferences::SetGizmoOperation(int operation)
	{
		s_GizmoOperation = operation;
	}
}