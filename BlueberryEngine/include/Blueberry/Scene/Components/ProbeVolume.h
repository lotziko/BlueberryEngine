#pragma once

#include "Blueberry\Scene\Components\Component.h"

namespace Blueberry
{
	class BB_API ProbeVolume : public Component
	{
		OBJECT_DECLARATION(ProbeVolume)

	public:
		ProbeVolume() = default;
		virtual ~ProbeVolume() = default;

		const AABB& GetBounds();
		void SetBounds(const AABB& bounds);

		const Vector3Int& GetSize();
		void SetSize(const Vector3Int& size);

	private:
		AABB m_Bounds;
		Vector3Int m_Size;
	};
}