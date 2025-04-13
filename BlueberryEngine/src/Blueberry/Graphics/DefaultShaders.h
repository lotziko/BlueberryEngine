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
		static inline Shader* s_SkyboxShader = nullptr;
	};
}