#pragma once
#include "Editor\Assets\AssetImporter.h"

namespace Blueberry
{
	class AudioImporter : public AssetImporter
	{
		OBJECT_DECLARATION(AudioImporter)

	public:
		AudioImporter() = default;

		static String GetAudioPath(const Guid& guid);

	protected:
		virtual void ImportData() final;
	};
}