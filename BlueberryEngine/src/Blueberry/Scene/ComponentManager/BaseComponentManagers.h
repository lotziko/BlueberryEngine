#pragma once

#include "Blueberry\Core\Object.h"
#include "Blueberry\Scene\EnityComponent.h"
#include "ComponentManager.h"

class TransformManager : public TComponentManager<Transform> { };
class SpriteRendererManager : public TComponentManager<SpriteRenderer> { };
class CameraManager : public TComponentManager<Camera> { };