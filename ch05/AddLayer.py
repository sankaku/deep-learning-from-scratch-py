# forward/backward propagation of addition layer


class AddLayer:
    """
    Addition layer

    This class does not have fields unlike `MultiplyLayer`
    because `backward` method does not use them.
    """

    def __init__(self):
        pass  # do nothing. no fields.

    def forward(self, x, y):
        return x + y

    def backward(self, doutdz):
        """
        calculate backpropagation of `x + y = z`.
        d(out)/dz -> d(out)/dx = d(out)/dz * dz/dx = d(out)/dz * 1
        d(out)/dz -> d(out)/dy = d(out)/dz * dz/dy = d(out)/dz * 1
        doutdz: d(out)/dz. This is forward output of this node.
        """
        doutdx = doutdz * 1
        doutdy = doutdz * 1
        return doutdx, doutdx


def show_forward(add_layer):
    """
    Example.

    How much does it cost to buy an apple(100 yen), an orange(150 yen) and a kiwi(80 yen)?    
    """
    apple_price = 100
    orange_price = 150
    kiwi_price = 80

    apple_orange_price = add_layer.forward(apple_price, orange_price)
    total_price = add_layer.forward(apple_orange_price, kiwi_price)

    print('total_price = {0}'.format(total_price))


def show_backward(add_layer):
    dprice = 1
    dapple_orange_price, dkiwi_price = add_layer.backward(dprice)
    dapple_price, dorange_price = add_layer.backward(dapple_orange_price)

    print('dapple_price = {0}, dorange_price = {1}, dkiwi_price = {2}'.format(
        dapple_price, dorange_price, dkiwi_price))


if __name__ == '__main__':
    add_layer = AddLayer()
    show_forward(add_layer)
    show_backward(add_layer)
