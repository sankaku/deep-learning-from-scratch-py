# Example of using AddLayer and MultiplyLayer

# How much does it cost to buy 2 apples and 3 oranges?
# The price of an apple is 100 yen and that of an orange is 150 yen.
# Consumption tax is 110 %.

from MultiplyLayer import MultiplyLayer
from AddLayer import AddLayer

# prepare
add_layer = AddLayer()
apple_layer = MultiplyLayer()
orange_layer = MultiplyLayer()
tax_layer = MultiplyLayer()

apple_price = 100
apple_num = 2
orange_price = 150
orange_num = 3
tax_rate = 1.1

# forward
apples_price = apple_layer.forward(apple_price, apple_num)
oranges_price = orange_layer.forward(orange_price, orange_num)
apples_oranges_price = add_layer.forward(apples_price, oranges_price)
total_price = tax_layer.forward(apples_oranges_price, tax_rate)
print('total_price = {0}'.format(total_price))

# backward
dprice = 1
dapples_oranges_price, dtax = tax_layer.backward(dprice)
dapples_price, doranges_price = add_layer.backward(dapples_oranges_price)
dapple_price, dapple_num = apple_layer.backward(dapples_price)
dorange_price, dorange_num = orange_layer.backward(doranges_price)

print('dapple_price = {0}, dapple_num = {1}'.format(dapple_price, dapple_num))
print('dorange_price = {0}, dorange_num = {1}'.format(
    dorange_price, dorange_num))
print('dtax = {0}'.format(dtax))
