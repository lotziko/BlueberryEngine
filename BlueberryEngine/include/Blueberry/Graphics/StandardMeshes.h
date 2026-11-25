#pragma once

namespace Blueberry
{
	class Mesh;

	class StandardMeshes
	{
	public:
		static void Initialize();
		static Mesh* GetFullscreen();
		static Mesh* GetPlane();
		static Mesh* GetSphere();
		static Mesh* GetCube();
		static Mesh* GetCone();
		static Mesh* GetCylinder();

	private:
		static Mesh* s_FullscreenMesh;
		static Mesh* s_BlitMesh;
		static Mesh* s_PlaneMesh;
		static Mesh* s_SphereMesh;
		static Mesh* s_CubeMesh;
		static Mesh* s_ConeMesh;
		static Mesh* s_CylinderMesh;
	};
}