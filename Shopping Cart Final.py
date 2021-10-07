#This is Steps 1 thru 3 for Milestone Portfolio Option 1
class ItemToPurchase:
    def __int__(self, name, quantity, price):
        self.name = name
        self.quantity = quantity
        self.price = price
    def item_cost(self):
        print(self.name, self.quantity, "@", '$',(format(self.price, '.2f')))
    def item_total(self):
        print("The Total amount is ", total)
    

#User will now enter items to purchase from the store needed for baking

First_Item = ItemToPurchase()
First_Item.name = input(' Please enter the first item name = ')
First_Item.quantity = int(input(' Please enter the quantity needed = '))
First_Item.price = float(input( ' Please enter the price = $'))
Second_Item = ItemToPurchase()
Second_Item.name = input(' Please enter the second item name = ')
Second_Item.quantity = int(input(' Please enter the quantity needed = '))
Second_Item.price = float(input( ' Please enter the price = $'))
First_Item.item_cost = int()
Second_Item.item_cost = int()
#This is where items are totaled
First_Item_Total = float(First_Item.quantity * First_Item.price)
Second_Item_Total = float( Second_Item.quantity * Second_Item.price)
total =  format(First_Item_Total +  Second_Item_Total, '.2f')
print("The Total amount is $",(total))

#Below is the code for Steps 3 thru 10

class ItemToPurchase:

    def __init__(self, item_name='none', item_price=0, item_quantity=0, item_description = 'none'):
        self.item_name = item_name
        self.item_price = item_price
        self.item_quantity = item_quantity
        self.item_description = item_description


    def print_item_cost(self):
        string = '{} {} @ ${} = ${}'.format(self.item_name, self.item_quantity, self.item_price,
                                                float(self.item_quantity * self.item_price, '.2f'))
        cost = self.item_quantity * self.item_price
        return string, cost


    def print_item_description(self):
        string = '{}: {}'.format(self.item_name, self.item_description)
        print(string, end='\n')
        return string


class ShoppingCart:
    

    def __init__(self, customer_name = 'none', current_date = 'January 1, 2016', cart_items = []):
        self.customer_name = customer_name
        self.current_date = current_date
        self.cart_items = cart_items
    

    def add_item(self, string):

        print('\nAdd Item To Cart', end='\n')
       
        item_name = str(input('Please enter the item name: '))
        item_description = str(input('Please enter the item description: '))
        item_price = float(input('Please enter the item price: $'))
        item_quantity = int(input('Please enter the item quantity: '))      
        self.cart_items.append(ItemToPurchase(item_name, item_price, item_quantity, item_description))
    

    def remove_item(self):
        print('\nRemove Item From Cart', end='\n')     
        string = str(input('Enter name of item to be removed: '))
        i = 0     
        for item in self.cart_items:
            if(item.item_name == string):               
                del self.cart_items[i]             
                flag=True
                break           
            else:
                flag=False
                i += 1
        if(flag==False):           
            print('Item not found in cart. Nothing removed')

    
    def modify_item(self):
        print('\nChange Item Quantity', end='\n')       
        name = str(input('Enter the item name: '))               
        for item in self.cart_items:           
            if(item.item_name == name):
                quantity = int(input('Enter the new quantity: '))
                item.item_quantity = quantity               
                flag=True
                break
            else:
                flag=False
        if(flag==False):
            print('Item not found in cart. Nothing modified')


    def get_num_items_in_cart(self):
        num_items=0      
        for item in self.cart_items:
            num_items= num_items+item.item_quantity
        return num_items


    def get_cost_of_cart(self):
        total_cost = 0
        cost = 0
        for item in self.cart_items:
            cost = format(item.item_quantity * item.item_price, '.2f')
            total_cost += cost
        return total_cost

    def print_total():
        total_cost = get_cost_of_cart()
        if (total_cost == 0):
            print('Shopping Cart Is Empty')
        else:
            output_cart()
  

    def print_descriptions(self):
        print('\nOutput Items\' Descriptions')
        print('{}\'s Shopping Cart - {}'.format(self.customer_name, self.current_date),end='\n')
        print('\nItem Descriptions', end='\n')
        for item in self.cart_items:
            print('{}: {}'.format(item.item_name, item.item_description), end='\n')

    def output_cart(self):
        new=ShoppingCart()
        print('\nOutput Shopping Cart', end='\n')
        print('{}\'s Shopping Cart - {}'.format(self.customer_name, self.current_date),end='\n')      
        print('Number of Items:', new.get_num_items_in_cart(), end='\n\n')
        tc = 0
        for item in self.cart_items:
            print('{} {} @ ${} = ${}'.format(item.item_name, item.item_quantity,
                                             item.item_price, format(item.item_quantity * item.item_price, '.2f')),\
                  end='\n')
            tc += (item.item_quantity * item.item_price)
        print('\nTotal: ${}'.format(tc,'.2f'), end='\n')

def print_menu(ShoppingCart):
    customer_Cart = newCart
    string=''
    menu = ('\nMENU\n'
    'a - Add item to cart\n'
    'r - Remove item from cart\n'
    'c - Change item quantity\n'
    'i - Output items\' descriptions\n'
    'o - Output shopping cart\n'
    'q - Quit\n')
    command = ''
    while(command != 'q'):
        string=''
        print(menu, end='\n')
        command = input('Choose an option: ')
        while(command != 'a' and command != 'o' and command != 'i' and command != 'r'
              and command != 'c' and command != 'q'):
            command = input('Choose an option: ')
        if(command == 'a'):
            customer_Cart.add_item(string)
        if(command == 'o'):
            customer_Cart.output_cart()
        if(command == 'i'):
            customer_Cart.print_descriptions()
        if(command == 'r'):
            customer_Cart.remove_item()
        if(command == 'c'):
            customer_Cart.modify_item()

customer_name = str(input('Enter customer\'s name: '))
current_date = str(input('Enter today\'s date: '))
print('\nCustomer name:', customer_name, end='\n')
print('Today\'s date:', current_date, end='\n')
newCart = ShoppingCart(customer_name, current_date)
print_menu(newCart)