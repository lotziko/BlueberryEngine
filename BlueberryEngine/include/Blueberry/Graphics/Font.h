#pragma once

#include "Blueberry\Core\Object.h"

namespace Blueberry
{
	class BB_API Font : public Object
	{
		OBJECT_DECLARATION(Font)

	public:
		Font() = default;
		virtual ~Font() = default;

		const ByteData& GetData() const;
		void SetData(const ByteData& data);

	private:
		ByteData m_RawData = {};
	};
}