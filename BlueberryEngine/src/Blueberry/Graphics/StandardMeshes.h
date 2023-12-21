#pragma once

namespace Blueberry
{
	class Mesh;

	class StandardMeshes
	{
	public:
		static Mesh* GetFullscreen();

	private:
		static Mesh* s_FullscreenMesh;
	};
}