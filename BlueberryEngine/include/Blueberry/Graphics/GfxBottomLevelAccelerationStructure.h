#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Mesh;
	class Material;

	class GfxBottomLevelAccelerationStructure
	{
	public:
		BB_OVERRIDE_NEW_DELETE

		virtual ~GfxBottomLevelAccelerationStructure() = default;

		static GfxBottomLevelAccelerationStructure* Get(Mesh* mesh, bool opaqueMask[16]);

	private:
		static GfxBottomLevelAccelerationStructure* CreateAccelerationStructure(Mesh* mesh, bool opaqueMask[16]);

	private:
		static Dictionary<ObjectId, std::pair<uint32_t, GfxBottomLevelAccelerationStructure*>> s_AccelerationStructures;
	};
}