#pragma once

template<class Type>
class Hierarchy 
{
public:
	~Hierarchy();

	void SetParent(Hierarchy<Type>& node);
	void MakeSiblingAfter(Hierarchy<Type>& node);
	bool ParentedBy(const Hierarchy<Type>& node) const;
	void RemoveFromParent();
	void RemoveFromHierarchy();
	void SetOwner(Type* newOwner);
	Type* GetOwner() const;

	Type* GetParent() const;
	Type* GetChild() const;
	Type* GetSibling() const;
	Type* GetPriorSibling() const;
	Type* GetNext() const;
	Type* GetNextLeaf() const;

private:
	Hierarchy<Type>* GetPriorSiblingNode() const;

private:
	Hierarchy<Type>* m_Parent = nullptr;
	Hierarchy<Type>* m_Sibling = nullptr;
	Hierarchy<Type>* m_Child = nullptr;
	Type* m_Owner = nullptr;
};

//*******************
// Hierarchy::~Hierarchy
//*******************
template<class Type>
inline Hierarchy<Type>::~Hierarchy() {
	RemoveFromHierarchy();
}

//*******************
// Hierarchy::ParentedBy
// returns true if param node is an ancestor of *this
// returns false otherwise
//*******************
template<class Type>
inline bool Hierarchy<Type>::ParentedBy(const Hierarchy<Type>& node) const 
{
	if (m_Parent == &node) 
	{
		return true;
	}
	else if (m_Parent != nullptr) 
	{
		return m_Parent->ParentedBy(node);
	}
	return false;
}

//*******************
// Hierarchy::SetParent
// sets *this as the first m_Child of param node
// and param node's first m_Child as the first m_Sibling of *this
//*******************
template<class Type>
inline void Hierarchy<Type>::SetParent(Hierarchy<Type>& node) 
{
	RemoveFromParent();
	m_Parent = &node;
	m_Sibling = node.m_Child;
	node.m_Child = this;
}

//*******************
// Hierarchy::MakeSiblingAfter
//*******************
template<class Type>
inline void Hierarchy<Type>::MakeSiblingAfter(Hierarchy<Type>& node) 
{
	RemoveFromParent();
	m_Parent = node.m_Parent;
	m_Sibling = node.m_Sibling;
	node.m_Sibling = this;
}

//*******************
// Hierarchy::RemoveFromParent
// makes prior m_Sibling the new m_Child of m_Parent
// then nullifies m_Parent and m_Sibling, but
// retains the m_Child pointer, effectively making
// this a top-level m_Parent node
//*******************
template<class Type>
inline void Hierarchy<Type>::RemoveFromParent() 
{
	Hierarchy<Type>* prev = GetPriorSiblingNode();

	if (prev != nullptr) 
	{
		prev->m_Sibling = m_Sibling;
	}
	else if (m_Parent != nullptr) 
	{
		m_Parent->m_Child = m_Sibling;
	}

	m_Parent = nullptr;
	m_Sibling = nullptr;
}

//*******************
// Hierarchy::RemoveFromHierarchy
// removes *this from the hierarchy
// and adds it's children it's m_Parent, 
// or makes the children's m_Parent nullptr (a top-level set of siblings)
//*******************
template<class Type>
inline void Hierarchy<Type>::RemoveFromHierarchy() 
{
	Hierarchy<Type>* parentNode = m_Parent;
	Hierarchy<Type>* oldChild = nullptr;

	RemoveFromParent();
	if (parentNode != nullptr) 
	{
		while (m_Child != nullptr) 
		{
			oldChild = m_Child;
			oldChild->RemoveFromParent();
			oldChild->SetParent(parentNode);
		}
	}
	else 
	{
		while (m_Child != nullptr)
			m_Child->RemoveFromParent();
	}
}

//*******************
// Hierarchy::SetOwner
// sets the object this node is associated with
//*******************
template<class Type>
inline void Hierarchy<Type>::SetOwner(Type* newOwner) 
{
	m_Owner = newOwner;
}

//*******************
// Hierarchy::GetOwner
// returns the object associated with this node
//*******************
template<class Type>
inline Type* Hierarchy<Type>::GetOwner() const 
{
	return m_Owner;
}

//*******************
// Hierarchy::GetParent
// returns the common node among siblings
//*******************
template<class Type>
inline Type* Hierarchy<Type>::GetParent() const 
{
	return m_Parent->m_Owner;
}

//*******************
// Hierarchy::GetChild
// returns the only m_Child of this node
//*******************
template<class Type>
inline Type* Hierarchy<Type>::GetChild() const 
{
	return m_Child->m_Owner;
}

//*******************
// Hierarchy::GetSibling
// returns the next node with the same m_Parent
//*******************
template<class Type>
inline Type* Hierarchy<Type>::GetSibling() const 
{
	return m_Sibling->m_Owner;
}

//*******************
// Hierarchy::GetPriorSibling
// returns the m_Owner of the previous node with the same m_Parent
//*******************
template<class Type>
inline Type* Hierarchy<Type>::GetPriorSibling() const 
{
	Hierarchy<Type>* prev = GetPriorSiblingNode();

	if (prev != nullptr) 
	{
		return prev->m_Owner;
	}

	return nullptr;
}

//*******************
// Hierarchy::GetNext
// traverses all nodes of the hierarchy, depth-first
// starting from *this
//*******************
template<class Type>
inline Type* Hierarchy<Type>::GetNext() const 
{
	if (m_Child != nullptr) 
	{
		return m_Child->m_Owner;
	}
	else if (m_Sibling != nullptr) 
	{
		return m_Sibling->m_Owner;
	}
	else
	{
		const Hierarchy<Type>* parentNode = m_Parent;
		while (parentNode != nullptr && parentNode->m_Sibling == nullptr) 
		{
			parentNode = parentNode->m_Parent;
		}

		if (parentNode != nullptr)
			return parentNode->m_Sibling->m_Owner;
	}

	return nullptr;
}

//*******************
// Hierarchy::GetNextLeaf
// traverses all leaf nodes of the hierarchy
// starting from *this
//*******************
template<class Type>
inline Type* Hierarchy<Type>::GetNextLeaf() const 
{
	// if there is no m_Child or m_Sibling, go up until a m_Parent wth a m_Sibling, then go down its children to a leaf

	const Hierarchy<Type>* node;

	if (m_Child != nullptr) 
	{
		node = m_Child;

		// not a leaf, so go down along m_Child and return the node w/m_Child == nullptr
		while (node->m_Child != nullptr) 
		{
			node = node->m_Child;
		}

		return node->m_Owner;
	}
	else 
	{
		node = this;

		// *this is a leaf, return its neighbor leaf
		while (node != nullptr && node->m_Sibling == nullptr) 
		{
			node = node->m_Parent;
		}

		if (node != nullptr) 
		{
			node = node->m_Sibling;

			// not a leaf, so go down along m_Child and return the node w/m_Child == nullptr
			while (node->m_Child != nullptr) 
			{
				node = node->m_Child;
			}

			return node->m_Owner;
		}
	}

	return nullptr;
}

//*******************
// Hierarchy::GetPriorSiblingNode
// returns previous node with the same m_Parent
// returns nullptr if *this is the first m_Child, or m_Parent is nullptr,
// as well as if *this is not a registered m_Child of its m_Parent
//*******************
template<class Type>
inline Hierarchy<Type>* Hierarchy<Type>::GetPriorSiblingNode() const 
{
	if (m_Parent != nullptr) 
	{
		Hierarchy<Type>* prev = m_Parent->m_Child;

		while (prev != nullptr && prev != this) 
		{
			if (prev->m_Sibling == this) 
			{
				return prev;
			}

			prev = prev->m_Sibling;
		}

		if (prev == nullptr) 
		{
			/*const std::string message("Hierarchy: node not a registered m_Child of its m_Parent.");
			eErrorLogger::LogError(message.c_str(), __FILE__, __LINE__);
			eErrorLogger::ErrorPopupWindow(message.c_str());*/
		}
	}

	return nullptr;
}
