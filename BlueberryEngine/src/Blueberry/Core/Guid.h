#pragma once

namespace Blueberry
{
	struct Guid
	{
	public:
		Guid() = default;

		bool operator==(const Guid &other) const;
		bool operator!=(const Guid &other) const;
		bool operator<(const Guid &other) const;

		std::string ToString() const;

		static Guid Create();
		void Generate();

	public:
		uint64_t data[2];
	};
}