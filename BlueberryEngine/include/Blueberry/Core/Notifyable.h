#pragma once

namespace Blueberry
{
	class Notifyable
	{
	public:
		virtual void OnNotify() = 0;
	};
}