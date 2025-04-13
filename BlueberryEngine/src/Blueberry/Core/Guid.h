#pragma once

namespace Blueberry
{
	using FileId = uint64_t;

	struct Guid
	{
	public:
		Guid() = default;
		Guid(const uint64_t& data1, const uint64_t& data2);

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

// Based on https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
// and https://stackoverflow.com/questions/37152892/existing-hash-function-for-uuid-t
template <>
struct std::hash<Blueberry::Guid>
{
	size_t operator()(const Blueberry::Guid& guid) const
	{
		return guid.data[0] ^ guid.data[1];
	}
};