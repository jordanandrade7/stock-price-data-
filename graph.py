import project1
import matplotlib.pyplot as plt

stock = project1.StockLibrary()
def longest():
    for i in stock.stockList:
        largest = [0].sname
        temp = [i + 1].sname
        index_counter = 0
        if len(largest) > len(temp):
            largest = i.sname
            index_counter += 1
        else:
            largest = temp
            index_counter += 1
        return largest, index_counter

        # x = np.linspace(0, 19, 19)
        # plt.plot(x, [index_counter].prices)
        # plt.title("Price Variations")
        # plt.xlabel("x")
        # plt.ylabel("Prices")
        # plt.show()

print(longest())