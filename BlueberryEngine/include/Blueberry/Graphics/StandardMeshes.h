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
		static Mesh* GetCube();

	private:
		static Mesh* s_FullscreenMesh;
		static Mesh* s_BlitMesh;
		static Mesh* s_PlaneMesh;
		static Mesh* s_SphereMesh;
		static Mesh* s_CubeMesh;
	};
}