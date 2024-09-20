#pragma once

namespace Blueberry
{
	class Mesh;

	class StandardMeshes
	{
	public:
		static Mesh* GetFullscreen();
		static Mesh* GetPlane();
		static Mesh* GetSphere();

	private:
		static inline Mesh* s_FullscreenMesh = nullptr;
		static inline Mesh* s_PlaneMesh = nullptr;
		static inline Mesh* s_SphereMesh = nullptr;
	};
}