package datago

// Define a set type using a map with empty struct values
type set map[string]struct{}

// Add an element to the set
func (s set) Add(value string) {
	s[value] = struct{}{}
}

// Remove an element from the set
func (s set) Remove(value string) {
	delete(s, value)
}

// Check if the set contains an element
func (s set) Contains(value string) bool {
	_, exists := s[value]
	return exists
}

// Get the size of the set
func (s set) Size() int {
	return len(s)
}
