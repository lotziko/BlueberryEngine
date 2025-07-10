#pragma once

#include <fstream>

namespace Blueberry
{
	class Mesh;

	class PhysicsShapeCache
	{
	public:
		static void* GetShape(Mesh* mesh);
		static void Bake(Mesh* mesh, std::ofstream& stream);
		static void Load(Mesh* mesh, std::ifstream& stream);
	};
}

