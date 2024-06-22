#pragma once

namespace Blueberry
{
	class Material;

	class DefaultMaterials
	{
	public:
		static Material* GetError();
		static Material* GetBlit();

	private:
		static inline Material* s_ErrorMaterial = nullptr;
		static inline Material* s_BlitMaterial = nullptr;
	};
}