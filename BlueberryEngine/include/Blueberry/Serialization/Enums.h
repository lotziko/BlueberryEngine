#pragma once

namespace Blueberry
{
	enum class SerializationFlags
	{
		None = 0,
		Text = 1,
		HasHeaders = 2, // YAML only
		HasGuids = 4, // Binary only

		EditorOnly = 8,
		RuntimeOnly = 16,
		EditorAndRuntime = 24,
	};

	inline SerializationFlags operator|(SerializationFlags lhs, SerializationFlags rhs)
	{
		return static_cast<SerializationFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
	}

	inline SerializationFlags operator&(SerializationFlags lhs, SerializationFlags rhs)
	{
		return static_cast<SerializationFlags>(static_cast<int>(lhs) & static_cast<int>(rhs));
	}
}