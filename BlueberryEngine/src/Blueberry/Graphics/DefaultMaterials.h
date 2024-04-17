#pragma once

namespace Blueberry
{
	class Material;

	class DefaultMaterials
	{
	public:
		static Material* GetError();

	private:
		static inline Material* s_ErrorMaterial = nullptr;
	};
}