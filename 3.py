print("Supermarket\n***************")

prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77] # Prices of products 1-10

total_price = 0  # Keeps track of the total amount

try:
    while True:
        # Ask the user to select a product or quit
        selected_product = int(input("Please select product (1-10) or 0 to quit: "))
        
        if 10 >= selected_product > 0:  # Valid product numbers
            total_price += prices[selected_product - 1]
            print(f"Product {selected_product} Price: {prices[selected_product - 1]}")
        elif selected_product == 0:  # Quit and process payment
            print(f"Total: {total_price}")
            while True:
                payment = int(input("Payment: "))
                if payment < total_price:  # Check if payment is sufficient
                    print(f"Insufficient payment. Please enter at least {total_price}.")
                else:
                    print(f"Payment: {payment} \nChange: {payment - total_price}")
                    break
            break
        else:
            print("Invalid product selection. Please select 1-10.")
except ValueError:
    print("Please enter a valid number.")
