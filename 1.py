shopping_list = []

while True:
    choice = input(f"Would you like to\n(1)Add or \n(2)Remove items or \n(3)Quit?: ")

    if choice == '1':
        new_item = input("What will be added?: ")
        shopping_list.append(new_item)

    elif choice == '2':
        if len(shopping_list) == 0:
            print("The list is empty.")
        else:
            print(f"There are {len(shopping_list)} items in the list.")
            remove_index_str = input("Which item is deleted?: ")
            if remove_index_str.isdigit() and 1 <= int(remove_index_str) <= len(shopping_list):
                remove_index = int(remove_index_str) - 1
                del shopping_list[remove_index]
            else:
                print("Incorrect selection.")

    elif choice == '3':
        print("The following items remain in the list:")
        for item in shopping_list:
            print(item)
        break

    else:
        print("Incorrect selection.")