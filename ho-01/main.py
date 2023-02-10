input = open("../datasets/Shakespeare.txt")
output = open("Shakespeare_normalized.txt", "w+")
text = input.read()
text = text.lower()
for char in text:
  if (not char.isalnum()):
    text = text.replace(char, " ")

output.write(text)