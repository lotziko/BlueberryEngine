#pragma once

namespace Blueberry
{
	class Notifyable
	{
	public:
		virtual void OnNotify(void* args) = 0;
	};
}