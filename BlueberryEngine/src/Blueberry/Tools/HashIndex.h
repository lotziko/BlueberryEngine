#pragma once

class HashIndex
{
public:
	HashIndex();
	explicit HashIndex(int initialHashSize);

	template <class ForwardIterator, class HashFunction>
	HashIndex(ForwardIterator begin, ForwardIterator end, HashFunction hasher);

	template <class ForwardIterator, class HashFunction>
	void Rebuild(ForwardIterator begin, ForwardIterator end, HashFunction hasher);

	void Add(const int hashkey, const int index);
	void Remove(const int hashkey, const int index);
	int First(const int hashkey) const;
	int Next(const int index) const;

	void InsertIndex(const int hashkey, const int index);
	void RemoveIndex(const int hashkey, const int index);
	void Clear();
	void ClearAndResize(int newHashSize);
	int NumUniqueKeys() const;
	int NumDuplicateKeys() const;
	int GetSpread() const;

	size_t HashCapacity() const;
	size_t IndexCapacity() const;

private:

	std::vector<int> m_Hash;
	std::vector<int> m_IndexChain;
	int m_HashMask;

	static const int DEFAULT_HASH_SIZE = 1024;
	static const int INVALID_INDEX = -1;
};

inline HashIndex::HashIndex() : HashIndex(DEFAULT_HASH_SIZE)
{
}

inline HashIndex::HashIndex(int initialHashSize)
{
	int power = 0;
	while (initialHashSize >> ++power);
	initialHashSize = 1 << power;

	m_Hash.reserve(initialHashSize);
	m_IndexChain.reserve(initialHashSize);
	m_Hash.assign(initialHashSize, INVALID_INDEX);
	m_IndexChain.assign(initialHashSize, INVALID_INDEX);
	m_HashMask = m_Hash.size() - 1;
}

template <class ForwardIterator, class HashFunction>
inline HashIndex::HashIndex(ForwardIterator begin, ForwardIterator end, HashFunction hasher) : HashIndex(std::distance(begin, end))
{
	for (ForwardIterator i = begin; i != end; i++)
	{
		Add(hasher(*i), std::distance(begin, i));
	}
}

template <class ForwardIterator, class HashFunction>
inline void HashIndex::Rebuild(ForwardIterator begin, ForwardIterator end, HashFunction hasher)
{
	ClearAndResize(std::distance(begin, end));
	for (ForwardIterator i = begin; i != end; i++)
	{
		Add(hasher(*i), std::distance(begin, i));
	}
}

inline void HashIndex::Add(const int hashkey, const int index)
{
	if (index >= m_IndexChain.size())     // DEBUG: std::vector may allocate more than max-signed-int values, but not for my purposes
		m_IndexChain.resize(index + 1);   // DEBUG: m_IndexChain.size() need not be a power of 2, it doesn't affect hashkey spread

	int k = hashkey & m_HashMask;
	m_IndexChain[index] = m_Hash[k];
	m_Hash[k] = index;
}

inline void HashIndex::Remove(const int hashkey, const int index)
{
	int k;

	if (m_Hash.empty())
		return;

	k = hashkey & m_HashMask;
	if (m_Hash[k] == index)
	{
		m_Hash[k] = m_IndexChain[index];
	}
	else
	{
		for (int i = m_Hash[k]; i != INVALID_INDEX; i = m_IndexChain[i])
		{
			if (m_IndexChain[i] == index)
			{
				m_IndexChain[i] = m_IndexChain[index];
				break;
			}
		}
	}
	m_IndexChain[index] = INVALID_INDEX;
}

inline int HashIndex::First(const int hashkey) const
{
	return m_Hash[hashkey & m_HashMask];
}

inline int HashIndex::Next(const int index) const
{
	return m_IndexChain[index];
}

inline void HashIndex::InsertIndex(const int hashkey, const int index)
{
	int max = index;
	for (size_t i = 0; i < m_Hash.size(); i++)
	{
		if (m_Hash[i] >= index)
		{
			m_Hash[i]++;
			if (m_Hash[i] > max)
			{
				max = m_Hash[i];
			}
		}
	}
	for (size_t i = 0; i < m_IndexChain.size(); i++)
	{
		if (m_IndexChain[i] >= index)
		{
			m_IndexChain[i]++;
			if (m_IndexChain[i] > max)
			{
				max = m_IndexChain[i];
			}
		}
	}
	if (max >= m_IndexChain.size())
	{
		m_IndexChain.resize(max + 1);     // DEBUG: m_IndexChain.size() need not be a power of 2, it doesn't affect hashkey spread
	}
	for (int i = max; i > index; i--)
	{
		m_IndexChain[i] = m_IndexChain[i - 1];
	}
	m_IndexChain[index] = INVALID_INDEX;
	Add(hashkey, index);
}

inline void HashIndex::RemoveIndex(const int hashkey, const int index)
{
	Remove(hashkey, index);
	int max = index;
	for (size_t i = 0; i < m_Hash.size(); i++)
	{
		if (m_Hash[i] >= index)
		{
			if (m_Hash[i] > max)
			{
				max = m_Hash[i];
			}
			m_Hash[i]--;
		}
	}
	for (size_t i = 0; i < m_IndexChain.size(); i++)
	{
		if (m_IndexChain[i] >= index)
		{
			if (m_IndexChain[i] > max)
			{
				max = m_IndexChain[i];
			}
			m_IndexChain[i]--;
		}
	}
	for (int i = index; i < max; i++)
	{
		m_IndexChain[i] = m_IndexChain[i + 1];
	}
	m_IndexChain[max] = INVALID_INDEX;
}

inline void HashIndex::Clear()
{
	m_Hash.assign(m_Hash.capacity(), INVALID_INDEX);
	m_IndexChain.assign(m_Hash.capacity(), INVALID_INDEX);
}

inline void HashIndex::ClearAndResize(int newHashSize)
{
	int power = 0;
	while (newHashSize >> ++power);
	newHashSize = 1 << power;

	m_Hash.resize(newHashSize);
	m_IndexChain.resize(newHashSize);
	m_Hash.assign(m_Hash.capacity(), INVALID_INDEX);
	m_IndexChain.assign(m_Hash.capacity(), INVALID_INDEX);
	m_HashMask = m_Hash.size() - 1;
}

inline int HashIndex::NumUniqueKeys() const
{
	int uniqueCount = 0;
	for (size_t i = 0; i < m_Hash.size(); i++)
	{
		if (m_Hash[i] != -1)
		{
			uniqueCount++;
		}
	}
	return uniqueCount;
}

inline int HashIndex::NumDuplicateKeys() const
{
	int duplicateCount = 0;
	for (size_t i = 0; i < m_IndexChain.size(); i++)
	{
		if (m_IndexChain[i] != -1)
		{
			duplicateCount++;
		}
	}
	return duplicateCount;
}

inline size_t HashIndex::HashCapacity() const
{
	return m_Hash.capacity();
}

inline size_t HashIndex::IndexCapacity() const
{
	return m_IndexChain.capacity();
}

inline int HashIndex::GetSpread() const
{
	int i, index, totalItems;
	std::vector<int> numHashItems;
	int average, error, e;

	const int hashSize = m_Hash.size();

	if (NumUniqueKeys() == 0)
	{
		return 100;
	}

	totalItems = 0;
	numHashItems.reserve(hashSize);
	numHashItems.assign(hashSize, 0);
	for (i = 0; i < hashSize; i++)
	{
		for (index = m_Hash[i]; index != INVALID_INDEX; index = m_IndexChain[index])
		{
			numHashItems[i]++;
		}
		totalItems += numHashItems[i];
	}
	// if no items in m_Hash
	if (totalItems <= 1)
	{
		return 100;
	}
	average = totalItems / hashSize;
	error = 0;
	for (i = 0; i < hashSize; i++)
	{
		e = abs(numHashItems[i] - average);
		if (e > 1)
		{
			error += e - 1;
		}
	}
	return 100 - (error * 100 / totalItems);
}