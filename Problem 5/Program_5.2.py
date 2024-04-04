def is_balanced(text):
  """
  Checks if the parentheses in a string are balanced.

  Args:
      text: The string to check.

  Returns:
      True if the parentheses are balanced, False otherwise.
  """
  stack = []
  for char in text:
    if char in "([{":
      stack.append(char)  # Push opening parentheses onto the stack
    elif char in ")]}":
      if not stack:  # Empty stack means no opening parenthesis for this closing one
        return False
      top = stack.pop()  # Pop the top element from the stack
      if not is_matching(top, char):  # Check if closing parenthesis matches opening one
        return False
  return not stack  # If the stack is empty after processing, all parentheses were balanced

def is_matching(opening, closing):
  """
  Checks if the parentheses are a matching pair.

  Args:
      opening: The opening parenthesis character.
      closing: The closing parenthesis character.

  Returns:
      True if the parentheses are a matching pair, False otherwise.
  """
  return (opening == "(" and closing == ")") or \
         (opening == "{" and closing == "}") or \
         (opening == "[" and closing == "]")

# Test cases with expected output
test_cases = [
  ("()(())", True),  # Balanced
  ("()()", True),     # Balanced
  (")(()", False),    # Not balanced (extra closing parenthesis) 
  ("((", False),       # Not balanced (missing closing parenthesis)
  ("text", True),     # No parentheses, considered balanced
]

with open("output_5.2.txt", "w") as output_file:  # Open output file in write mode
  for text, expected_output in test_cases:
    result = is_balanced(text)
    output_string = f"Text: '{text}', Expected Output: {expected_output}, Result: {result}\n"
    print(output_string)  # Print output to console
    output_file.write(output_string)  # Write output to file

print("Results also saved to output_5.2.txt")
