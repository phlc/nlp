
#Open Shakespeare file
input = open("../datasets/Shakespeare.txt")

#Open outpu file 
output = open("Shakespeare_normalized.txt", "w+")

#Read Shakespeare.txt
text = input.read()

#Change all characters to lower case
text = text.lower()

#Replace non alphanumeric characters with one blank space
for char in text:
  if (not char.isalnum()):
    text = text.replace(char, " ")

#Write normalized text to output file
output.write(text)