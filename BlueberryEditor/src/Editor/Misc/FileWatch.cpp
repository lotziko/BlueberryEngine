#include "FileWatch.h"

#include "Concrete\Windows\Misc\WindowsFileWatch.h"

namespace Blueberry
{
	FileWatch* FileWatch::Create(const String& path)
	{
		return new WindowsFileWatch(path);
	}
}