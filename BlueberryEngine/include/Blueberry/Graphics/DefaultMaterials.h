#pragma once

namespace Blueberry
{
	class Material;

	class DefaultMaterials
	{
	public:
		static Material* GetError();
		static Material* GetBlit();
		static Material* GetResolveMSAA();
		static Material* GetPostProcessing();
		static Material* GetVRMirrorView();

	private:
		static Material* s_ErrorMaterial;
		static Material* s_BlitMaterial;
		static Material* s_ResolveMSAAMaterial;
		static Material* s_PostProcessingMaterial;
		static Material* s_VRMirrorViewMaterial;
	};
}