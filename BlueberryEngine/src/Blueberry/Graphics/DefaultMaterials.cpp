#include "bbpch.h"
#include "DefaultMaterials.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	Material* DefaultMaterials::GetError()
	{
		if (s_ErrorMaterial == nullptr)
		{
			Shader* shader = (Shader*)AssetLoader::Load("assets/shaders/Error.shader");
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load error shader.")
					return false;
			}
			s_ErrorMaterial = Material::Create(shader);
		}
		return s_ErrorMaterial;
	}

	Material* DefaultMaterials::GetBlit()
	{
		if (s_BlitMaterial == nullptr)
		{
			Shader* shader = (Shader*)AssetLoader::Load("assets/shaders/Blit.shader");
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load blit shader.")
					return false;
			}
			s_BlitMaterial = Material::Create(shader);
		}
		return s_BlitMaterial;
	}
}
