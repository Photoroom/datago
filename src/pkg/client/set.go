package datago

// Define a set type using a map with empty struct values
type Set map[string]struct{}

// Add an element to the set
func (s Set) Add(value string) {
	s[value] = struct{}{}
}

// Remove an element from the set
func (s Set) Remove(value string) {
	delete(s, value)
}

// Check if the set contains an element
func (s Set) Contains(value string) bool {
	_, exists := s[value]
	return exists
}

// Get the size of the set
func (s Set) Size() int {
	return len(s)
}
