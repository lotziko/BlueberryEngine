#include "Blueberry\Graphics\DefaultMaterials.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	Material* DefaultMaterials::GetError()
	{
		if (s_ErrorMaterial == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/Error.shader"));
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load error shader.");
				return nullptr;
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
				BB_ERROR("Failed to load blit shader.");
				return nullptr;
			}
			s_BlitMaterial = Material::Create(shader);
			s_BlitMaterial->SetName("Blit");
		}
		return s_BlitMaterial;
	}

	Material* DefaultMaterials::GetResolveMSAA()
	{
		if (s_ResolveMSAAMaterial == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/ResolveMSAA.shader"));
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load resolve MSAA shader.");
				return nullptr;
			}
			s_ResolveMSAAMaterial = Material::Create(shader);
			s_ResolveMSAAMaterial->SetName("ResolveMSAA");
		}
		return s_ResolveMSAAMaterial;
	}

	Material* DefaultMaterials::GetPostProcessing()
	{
		if (s_PostProcessingMaterial == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/PostProcessing.shader"));
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load post processing shader.");
				return nullptr;
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
				BB_ERROR("Failed to load VR mirror view shader.");
				return nullptr;
			}
			s_VRMirrorViewMaterial = Material::Create(shader);
			s_VRMirrorViewMaterial->SetName("VRMirror");
		}
		return s_VRMirrorViewMaterial;
	}
}
