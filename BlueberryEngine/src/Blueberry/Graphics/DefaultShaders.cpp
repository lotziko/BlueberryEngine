#include "bbpch.h"
#include "DefaultShaders.h"

#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Assets\AssetLoader.h"
#include "Blueberry\Core\ObjectDB.h"

namespace Blueberry
{
	void DefaultShaders::Initialize()
	{
		GetSkybox();
	}

	Shader* DefaultShaders::GetSkybox()
	{
		if (s_SkyboxShader == nullptr)
		{
			Shader* shader = static_cast<Shader*>(AssetLoader::Load("assets/shaders/Skybox.shader"));
			if (shader == nullptr)
			{
				BB_ERROR("Failed to load skybox shader.")
					return false;
			}
			ObjectDB::AllocateIdToGuid(shader, Guid(1, 0), 10000);
			shader->SetName("Skybox");
			s_SkyboxShader = shader;
		}
		return s_SkyboxShader;
	}
}
