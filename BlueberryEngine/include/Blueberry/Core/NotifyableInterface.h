#pragma once

namespace Blueberry
{
	class NotifyableInterface
	{
	public:
		virtual void OnNotify(void* args) = 0;
	};
}