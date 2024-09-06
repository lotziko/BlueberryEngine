#include "Preferences.h"

float* Preferences::GetGizmoSnapping()
{
	return s_GizmoSnapping;
}

const int& Preferences::GetGizmoOperation()
{
	return s_GizmoOperation;
}

void Preferences::SetGizmoOperation(const int& operation)
{
	s_GizmoOperation = operation;
}
