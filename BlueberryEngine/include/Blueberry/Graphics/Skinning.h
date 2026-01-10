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
		static inline ComputeShader* s_SkinningShader = nullptr;
		static inline GfxBuffer* s_ConstantBuffer = nullptr;
		static inline GfxBuffer* s_BoneTransformBuffer = nullptr;
	};
}