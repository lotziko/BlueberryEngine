#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class ProbeVolume : public Component
	{
		OBJECT_DECLARATION(ProbeVolume)

	public:
		ProbeVolume() = default;
		virtual ~ProbeVolume() = default;

		virtual void OnEnable() final;
		virtual void OnDisable() final;

		const AABB& GetBounds();
		const Vector3Int& GetSize();

	private:
		AABB m_Bounds;
		Vector3Int m_Size;
	};
}