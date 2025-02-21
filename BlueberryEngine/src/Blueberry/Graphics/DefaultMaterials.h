#pragma once

namespace Blueberry
{
	class Material;

	class DefaultMaterials
	{
	public:
		static Material* GetError();
		static Material* GetBlit();
		static Material* GetVRMirrorView();
		static Material* GetSkybox();

	private:
		static inline Material* s_ErrorMaterial = nullptr;
		static inline Material* s_BlitMaterial = nullptr;
		static inline Material* s_VRMirrorViewMaterial = nullptr;
		static inline Material* s_SkyboxMaterial = nullptr;
	};
}