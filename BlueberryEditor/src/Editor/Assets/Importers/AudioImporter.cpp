#include "AudioImporter.h"

#include "Blueberry\Audio\AudioClip.h"
#include "Blueberry\Tools\FileHelper.h"

#include "Editor\Assets\AssetDB.h"

namespace Blueberry
{
	OBJECT_DEFINITION(AudioImporter, AssetImporter)
	{
		DEFINE_BASE_FIELDS(AudioImporter, AssetImporter)
	}

	String AudioImporter::GetAudioPath(const Guid& guid)
	{
		std::filesystem::path dataPath = Path::GetAudioCachePath();
		if (!std::filesystem::exists(dataPath))
		{
			std::filesystem::create_directories(dataPath);
		}
		dataPath.append(guid.ToString().append(".clip"));
		return StringHelper::ToString(dataPath);
	}

	void AudioImporter::ImportData()
	{
		Guid guid = GetGuid();
		List<Object*> objects;
		String audioPath = GetAudioPath(guid);
		String path = GetFilePath();

		// TODO compression
		List<uint8_t> data;
		FileHelper::Load(data, path);

		size_t audioClipFileId = TO_HASH("AudioClip");
		AudioClip* audioClip = GetOrCreateAssetObject<AudioClip>(audioClipFileId);
		SetMainObject(audioClipFileId);
		audioClip->SetName(GetName());
		audioClip->Initialize(data);

		objects.push_back(audioClip);

		AssetDB::SaveAssetObjectsToCache(List<Object*> { audioClip });
		FileHelper::Save(data, audioPath);
	}
}