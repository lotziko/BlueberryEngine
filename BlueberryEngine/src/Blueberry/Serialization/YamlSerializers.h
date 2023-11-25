#pragma once

#include "rapidyaml\ryml.h"
#include "Blueberry\Math\Math.h"
#include "Blueberry\Core\Guid.h"
#include "Blueberry\Tools\ByteConverter.h"

namespace DirectX::SimpleMath
{
	size_t to_chars(ryml::substr buf, Blueberry::Vector2 v);
	bool from_chars(ryml::csubstr buf, Blueberry::Vector2 *v);

	size_t to_chars(ryml::substr buf, Blueberry::Vector3 v);
	bool from_chars(ryml::csubstr buf, Blueberry::Vector3 *v);

	size_t to_chars(ryml::substr buf, Blueberry::Vector4 v);
	bool from_chars(ryml::csubstr buf, Blueberry::Vector4 *v);

	size_t to_chars(ryml::substr buf, Blueberry::Quaternion q);
	bool from_chars(ryml::csubstr buf, Blueberry::Quaternion *q);
}

namespace Blueberry
{
	size_t to_chars(ryml::substr buf, Blueberry::Guid val);
	bool from_chars(ryml::csubstr buf, Blueberry::Guid *v);
}