# forward/backward propagation of multiplication layer


class MultiplyLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, doutdz):
        """
        calculate backpropagation of `x * y = z`.

        d(out)/dz -> d(out)/dx = d(out)/dz * dz/dx = d(out)/dz * y
        d(out)/dz -> d(out)/dy = d(out)/dz * dz/dy = d(out)/dz * x
        doutdz: d(out)/dz. This is forward output of this node.
        """
        doutdx = doutdz * self.y
        doutdy = doutdz * self.x
        return doutdx, doutdy


def show_forward(apple_layer, tax_layer):
    """
    Example of forward propagation.

    How much does it cost to buy 2 apples?
    1. An apple costs 100 yen.
    2. Consumption tax is 110 %.

    apple_layer: MultiplyLayer for apple
    tax_layer: MultiplyLayer for tax
    """
    # buy 2 apples
    apples_price = apple_layer.forward(2, 100)
    # tax
    total_price = tax_layer.forward(apples_price, 1.1)
    print('total_price = {0}'.format(total_price))


def show_backward(apple_layer, tax_layer):
    """
    Example of backward propagation.

    This method must be called after `show_forward`
    because it initializes its fields(x, y).
    apple_layer: MultiplyLayer for apple
    tax_layer: MultiplyLayer for tax
    """
    dprice = 1

    dapples_price, dtax = tax_layer.backward(dprice)
    dapple_price, dapple_num = apple_layer.backward(dapples_price)
    print('dapple_price = {0}, dapple_num = {1}, dtax = {2}'.format(
        dapple_price, dapple_num, dtax))


if __name__ == '__main__':
    apple_layer = MultiplyLayer()
    tax_layer = MultiplyLayer()

    show_forward(apple_layer, tax_layer)
    show_backward(apple_layer, tax_layer)
