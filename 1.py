def main():
    while True:
        user_response = input("Write something (quit ends): ").lower()
        if user_response == "quit":
            break
        print(tester(user_response))


def tester(givenstring="Too short"):
    if len(givenstring) < 10:
        return "Too short"
    return givenstring


main()
