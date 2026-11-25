#include "Preferences.h"

namespace Blueberry
{
	float Preferences::s_GizmoSnapping[3] = { 0.1f, 5.0f, 1.0f };
	int Preferences::s_GizmoOperation = 0;
	bool Preferences::s_IsGizmoLocalSpace = false;

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

	bool Preferences::IsGizmoLocalSpace()
	{
		return s_IsGizmoLocalSpace;
	}

	void Preferences::SetGizmoLocalSpace(bool isLocalSpace)
	{
		s_IsGizmoLocalSpace = isLocalSpace;
	}
}