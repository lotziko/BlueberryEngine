#pragma once

namespace Blueberry
{
	class Serializer
	{
		virtual void Serialize() = 0;
		virtual void Deserialize() = 0;
	};
}