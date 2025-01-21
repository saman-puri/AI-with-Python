def my_split(sentence, separator): # Split sentence into words
    result = []
    current_word = ""
    for char in sentence:
        if char == separator: # when character is separator, the word is complete
            result.append(current_word)
            current_word = "" # Resetting the current word to start new word
        else:
            current_word += char
    result.append(current_word)  # Add the last word
    return result  #Return word list


def my_join(word_list, separator):
    result = ""
    for word in word_list:
        result += f"{word}\n" #Placing the result in new line
    return result

sentence = input("Please enter sentence: ")
separator = " "

print(my_split(sentence, separator))

print(my_join(my_split(sentence, separator), ","))
