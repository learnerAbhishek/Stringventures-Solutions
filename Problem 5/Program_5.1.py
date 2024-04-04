# Function to check if a string is a palindrome or can be converted into a palindrome by removing one character
def is_almost_palindrome(s):
    """
    Check if the provided string is either a palindrome or can be converted into a palindrome 
    by removing one character.
    
    Args:
    - s (str): The string to be checked.
    
    Returns:
    - bool: True if the string is an almost palindrome, False otherwise.
    """
    # Lambda function to check if a string is a palindrome
    is_palindrome = lambda x: x == x[::-1]

    if is_palindrome(s):
        return True

    # Try removing each character and check if the resulting string is a palindrome
    for i in range(len(s)):
        new_string = s[:i] + s[i+1:]
        if is_palindrome(new_string):
            return True

    return False

# Test cases
test_cases = [
    ("aaba", True),     # Removing one 'a' makes it a palindrome: "aba"
    ("racecar", True),  # Already a palindrome
    ("hello", False),   # Not a palindrome and cannot be converted into a palindrome by removing one character
    ("level", True),    # Already a palindrome
    ("radar", True),    # Already a palindrome
    ("abccdba", True),  # Removing one 'd' makes it a palindrome: "abccba"
    ("", True),         # Empty string is considered a palindrome
    ("a", True),        # Single character string is considered a palindrome
]

# Open output file for writing
with open("output_5.1.txt", "w") as f:
    # Run test cases and write results to output file
    for test_input, expected_output in test_cases:
        result = is_almost_palindrome(test_input)
        print(f"Input: {test_input}, Expected Output: {expected_output}, Result: {result}")
        f.write(f"Input: {test_input}, Expected Output: {expected_output}, Result: {result}\n")

print("Results also saved to output_5.1.txt")
