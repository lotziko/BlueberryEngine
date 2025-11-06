#pragma once

class Preferences
{
public:
	static float* GetGizmoSnapping();

	static const int& GetGizmoOperation();
	static void SetGizmoOperation(const int& operation);

private:
	static inline float s_GizmoSnapping[3] = { 0.1f, 45.0f, 1.0f };
	static inline int s_GizmoOperation = 7;
};