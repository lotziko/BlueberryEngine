#include "bbpch.h"
#include "Guid.h"

#include <random>

namespace Blueberry
{
	bool Guid::operator==(const Guid& other) const
	{
		return std::memcmp(this, &other, sizeof(GUID)) == 0;
	}

	bool Guid::operator!=(const Guid& other) const
	{
		return std::memcmp(this, &other, sizeof(GUID)) != 0;
	}

	bool Guid::operator<(const Guid &other) const
	{
		return (data[0] < other.data[0] || (data[0] == other.data[0] && data[1] < other.data[1]));
	}

	std::string Guid::ToString() const
	{
		std::stringstream stream;
		stream << std::hex << data[0] << data[1];
		return stream.str();
	}

	Guid Guid::Create()
	{
		Guid guid;
		guid.Generate();
		return guid;
	}

	void Guid::Generate()
	{
		static std::random_device rd;
		static std::mt19937 gen(rd());

		static std::uniform_int_distribution<unsigned long long> dis(
			(std::numeric_limits<std::uint64_t>::min)(),
			(std::numeric_limits<std::uint64_t>::max)()
		);

		data[0] = dis(gen);
		data[1] = dis(gen);
	}
}
