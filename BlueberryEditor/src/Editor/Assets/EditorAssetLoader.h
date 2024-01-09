#pragma once
#include "Blueberry\Assets\AssetLoader.h"

namespace Blueberry
{
	class EditorAssetLoader final : public AssetLoader
	{
	protected:
		virtual Object* LoadImpl(const Guid& guid) override;
		virtual Object* LoadImpl(const std::string& path) override;
	};
}