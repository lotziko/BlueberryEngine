#pragma once

namespace Blueberry
{
	class Shader;

	class DefaultShaders
	{
	public:
		static void Initialize();

		static Shader* GetSkybox();

	private:
		static Shader* s_SkyboxShader;
	};
}