#pragma once

#include "rapidyaml\ryml.h"
#include "Blueberry\Math\Math.h"
#include "Blueberry\Core\Guid.h"
#include "Blueberry\Tools\ByteConverter.h"

namespace DirectX::SimpleMath
{
	size_t to_chars(ryml::substr buf, Vector2 v);
	bool from_chars(ryml::csubstr buf, Vector2 *v);

	size_t to_chars(ryml::substr buf, Vector3 v);
	bool from_chars(ryml::csubstr buf, Vector3 *v);

	size_t to_chars(ryml::substr buf, Vector4 v);
	bool from_chars(ryml::csubstr buf, Vector4 *v);

	size_t to_chars(ryml::substr buf, Quaternion q);
	bool from_chars(ryml::csubstr buf, Quaternion *q);

	size_t to_chars(ryml::substr buf, Color c);
	bool from_chars(ryml::csubstr buf, Color *c);
}

namespace Blueberry
{
	size_t to_chars(ryml::substr buf, Blueberry::Guid val);
	bool from_chars(ryml::csubstr buf, Blueberry::Guid *v);
}