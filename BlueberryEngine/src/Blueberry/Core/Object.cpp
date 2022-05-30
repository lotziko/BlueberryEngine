#include "bbpch.h"
#include "Object.h"

const std::size_t Object::Type = std::hash<std::string>()(TO_STRING(Object));