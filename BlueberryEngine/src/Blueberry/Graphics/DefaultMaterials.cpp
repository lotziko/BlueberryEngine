#include "DefaultMaterials.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "..\Assets\AssetLoader.h"

namespace Blueberry
{
	Material* DefaultMaterials::GetError()
	{
		if (s_ErrorMaterial == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/Error.shader"));
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load error shader.")
					return false;
			}
			s_ErrorMaterial = Material::Create(shader);
			s_ErrorMaterial->SetName("Error");
		}
		return s_ErrorMaterial;
	}

	Material* DefaultMaterials::GetBlit()
	{
		if (s_BlitMaterial == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/Blit.shader"));
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load blit shader.")
					return false;
			}
			s_BlitMaterial = Material::Create(shader);
			s_BlitMaterial->SetName("Blit");
		}
		return s_BlitMaterial;
	}

	Material* DefaultMaterials::GetPostProcessing()
	{
		if (s_PostProcessingMaterial == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/PostProcessing.shader"));
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load post processing shader.")
					return false;
			}
			s_PostProcessingMaterial = Material::Create(shader);
			s_PostProcessingMaterial->SetName("PostProcessing");
		}
		return s_PostProcessingMaterial;
	}

	Material* DefaultMaterials::GetVRMirrorView()
	{
		if (s_VRMirrorViewMaterial == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/VRMirrorView.shader"));
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load VR mirror view shader.")
					return false;
			}
			s_VRMirrorViewMaterial = Material::Create(shader);
			s_VRMirrorViewMaterial->SetName("VRMirror");
		}
		return s_VRMirrorViewMaterial;
	}
}
