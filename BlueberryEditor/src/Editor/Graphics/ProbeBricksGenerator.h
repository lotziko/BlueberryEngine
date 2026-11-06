#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class ProbeVolume;

	class ProbeBricksGenerator
	{
	public:
		static void GenerateProbeVolumeBricks(ProbeVolume* probeVolume);
	};
}