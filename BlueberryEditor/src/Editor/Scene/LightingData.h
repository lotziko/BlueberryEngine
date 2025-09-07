#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Structs.h"

namespace Blueberry
{
	class Texture2D;
	class Scene;

	class BB_API LightingData : public Object
	{
		OBJECT_DECLARATION(LightingData)

	public:
		LightingData() = default;
		virtual ~LightingData() = default;

		Vector4* GetChartScaleOffset();
		void SetChartScaleOffset(const List<Vector4>& scaleOffset);

		void SetInstanceOffset(const Dictionary<ObjectId, uint32_t>& instanceOffset);

		Texture2D* GetLightmap();
		void SetLightmap(Texture2D* lightmap);
		
		void Apply(Scene* scene);

	private:
		ObjectPtr<Texture2D> m_Lightmap;
		ByteData m_ChartScaleOffset;
		ByteData m_ChartInstanceOffset;
	};
}