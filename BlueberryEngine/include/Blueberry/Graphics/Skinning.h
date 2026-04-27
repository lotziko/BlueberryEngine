#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class GfxBuffer;
	class SkinnedMeshRenderer;
	class ComputeShader;

	class BB_API Skinning
	{
	public:
		static void Initialize();
		static void Shutdown();

		static GfxBuffer* GetVertexBuffer(SkinnedMeshRenderer* renderer);

	private:
		static ComputeShader* s_SkinningShader;
		static GfxBuffer* s_ConstantBuffer;
		static GfxBuffer* s_BoneTransformBuffer;
	};
}