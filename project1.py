"""
UMass ECE 241 - Advanced Programming
Project #1   Fall 2021
project1.py - Sorting and Searching

"""
import matplotlib.pyplot as plt
import time
import random
#imported classes from lecture notes
class TreeNode:
    def __init__(self, key, val, left=None, right=None, parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent
        self.balanceFactor = 0 # sets the balance factor to zero

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self

    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild or self.leftChild

    def hasBothChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self, key, value, lc, rc):
        self.key = key
        self.payload = value
        self.leftChild = lc
        self.rightChild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self

#imported from lecture notes
class BinarySearchTree:

    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def put(self, key, val):
        if self.root:
            self._put(key, val, self.root)
        else:
            self.root = TreeNode(key, val)
        self.size = self.size + 1

    def _put(self, key, val, currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)
            else:
                currentNode.leftChild = TreeNode(key, val, parent=currentNode)
        else:
            if currentNode.hasRightChild():
                self._put(key, val, currentNode.rightChild)
            else:
                currentNode.rightChild = TreeNode(key, val, parent=currentNode)

    def __setitem__(self, k, v):
        self.put(k, v)

    def get(self, key):
        if self.root:
            res = self._get(key, self.root)
            if res:
                return res.payload
            else:
                return None
        else:
            return None

    def _get(self, key, currentNode):
        if not currentNode:
            return None
        elif currentNode.key == key:
            return currentNode
        elif key < currentNode.key:
            return self._get(key, currentNode.leftChild)
        else:
            return self._get(key, currentNode.rightChild)

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        if self._get(key, self.root):
            return True
        else:
            return False

#imported from lecture notes
class AvlTree(BinarySearchTree):
    def _put(self, key, val, currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)
            else:
                currentNode.leftChild = TreeNode(key, val, parent=currentNode)
                self.updateBalance(currentNode.leftChild)
        else:
            if currentNode.hasRightChild():
                self._put(key, val, currentNode.rightChild)
            else:
                currentNode.rightChild = TreeNode(key, val, parent=currentNode)
                self.updateBalance(currentNode.rightChild)

    def updateBalance(self, node):
        if node.balanceFactor > 1 or node.balanceFactor < -1:
            self.rebalance(node)
            return
        if node.parent != None:
            if node.isLeftChild():
                node.parent.balanceFactor += 1
            elif node.isRightChild():
                node.parent.balanceFactor -= 1

            if node.parent.balanceFactor != 0:
                self.updateBalance(node.parent)

    def rotateLeft(self, rotRoot):
        newRoot = rotRoot.rightChild
        rotRoot.rightChild = newRoot.leftChild
        if newRoot.leftChild != None:
            newRoot.leftChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isLeftChild():
                rotRoot.parent.leftChild = newRoot
            else:
                rotRoot.parent.rightChild = newRoot
        newRoot.leftChild = rotRoot
        rotRoot.parent = newRoot
        rotRoot.balanceFactor = rotRoot.balanceFactor + 1 - min(
            newRoot.balanceFactor, 0)
        newRoot.balanceFactor = newRoot.balanceFactor + 1 + max(
            rotRoot.balanceFactor, 0)

    def rotateRight(self, rotRoot):
        newRoot = rotRoot.leftChild

        rotRoot.leftChild = newRoot.rightChild
        if newRoot.rightChild != None:
            newRoot.rightChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isRightChild():
                rotRoot.parent.rightChild = newRoot
            else:
                rotRoot.parent.leftChild = newRoot
        newRoot.rightChild = rotRoot
        rotRoot.parent = newRoot
        rotRoot.balanceFactor = rotRoot.balanceFactor - 1 - max(newRoot.balanceFactor, 0)
        newRoot.balanceFactor = newRoot.balanceFactor - 1 + min(rotRoot.balanceFactor, 0)

    def rebalance(self, node):
        if node.balanceFactor < 0:
            if node.rightChild.balanceFactor > 0:
                self.rotateRight(node.rightChild)
                self.rotateLeft(node)
            else:
                self.rotateLeft(node)
        elif node.balanceFactor > 0:
            if node.leftChild.balanceFactor < 0:
                self.rotateLeft(node.leftChild)
                self.rotateRight(node)
            else:
                self.rotateRight(node)


"""
Stock class for stock objects
"""
class Stock:

    """
    Constructor to initialize the stock object
    """
    def __init__(self, sname, symbol, val, prices): #stock objects
        self.sname = sname
        self.symbol = symbol
        self.val = val
        self.prices = prices
        pass

    """
    return the stock information as a string, including name, symbol, 
    market value, and the price on the last day (2021-02-01). 
    For example, the string of the first stock should be returned as: 
    “name: Exxon Mobil Corporation; symbol: XOM; val: 384845.80; price:44.84”. 
    """
    def __str__(self): #returns the attributes of each stock as a string
        return "name: " + self.sname + "; " + "symbol: " + self.symbol + "; " + "val: " + str(self.val[:-1]) + "; " + "price:" + str(self.prices[-1])
        pass

"""
StockLibrary class to mange stock objects
"""

class StockLibrary:

    """
    Constructor to initialize the StockLibrary
    """
    def __init__(self): # library objects
        self.stockList = [] # initialize list
        self.bst = None
        self.size = 0 # initialize at zero
        self.isSorted = False # initially false, later turned to true if sorted
        pass

    def size(self):
        return self.size # returns size

    def isSorted(self):
        return self.isSorted # returns true or false

    """
    The loadData method takes the file name of the input dataset,
    and stores the data of stocks into the library. 
    Make sure the order of the stocks is the same as the one in the input file. 
    """

    def loadData(self, filename: str):
        file = open(filename, 'r') # opens file
        file.readline() # skips first line because it is a title
        lines = file.readlines() # turns all lines in the file into list
        l_sname = [] # initialize list for name, symbol, value, and prices
        l_symbol = []
        l_val = []
        l_prices = []

        for line in lines: # scans through list
            attributes = line.split('|') # splits each line at '|' to seperate objects
            l_sname.append(attributes[0]) # appends each item to its corresponding list
            l_symbol.append(attributes[1])
            l_val.append(attributes[2])
            l_prices.append(attributes[3:])
            self.size = self.size + 1 # adds to the size of the list

        for i in range(self.size):
            self.stockList.append(Stock(l_sname[i], l_symbol[i], l_val[i], l_prices[i])) # creates stock object in a list
        return self.stockList
        pass

    """
    The linearSearch method searches the stocks based on sname or symbol.
    It takes two arguments as the string for search and an attribute field 
    that we want to search (“name” or “symbol”). 
    It returns the details of the stock as described in __str__() function 
    or a “Stock not found” message when there is no match. 
    """
    def linearSearch(self, query: str, attribute: str):
        if attribute == "name": # if user inputs a name
            for i in self.stockList: # loops through the list to match the input value
                if i.sname == query:
                    return "name: " + i.sname + "; " + "symbol: " + i.symbol + "; " + "val: " + str(i.val[:-1]) + "; " + "price:" + str(i.prices[18])

        elif attribute == "symbol": # if user inputs a symbol
            for j in self.stockList: # loops through the list to match the input value
                if j.symbol == query:
                    return "name: " + j.sname + "; " + "symbol: " + j.symbol + "; " + "val: " + str(j.val[:-1]) + "; " + "price:" + str(j.prices[18])

        else:
            return "Entry not found" # if 'name' or 'symbol' is not inputted
        return "Stock not found" # if there is no matching name or symbol in the list

    """
    Sort the stockList using QuickSort algorithm based on the stock symbol.
    The sorted array should be stored in the same stockList.
    Remember to change the isSorted variable after sorted
    """
    def quickSort(self):
        self.quickSortHelper(self.stockList, 0, self.size - 1)
        self.isSorted = True # adjust the isSorted value to true
        pass

    def quickSortHelper(self, alist, first, last): # code taken from lecture
        if first < last:
            splitpoint = self.partition(alist, first, last)

            self.quickSortHelper(alist, first, splitpoint - 1)
            self.quickSortHelper(alist, splitpoint + 1, last)

    def partition(self, alist, first, last):
        pivotvalue = alist[first].symbol # added symbol because we are sorting based on the symbol

        leftmark = first + 1
        rightmark = last

        done = False
        while not done:

            while leftmark <= rightmark and alist[leftmark].symbol <= pivotvalue:
                leftmark = leftmark + 1

            while alist[rightmark].symbol >= pivotvalue and rightmark >= leftmark:
                rightmark = rightmark - 1

            if rightmark < leftmark:
                done = True
            else:
                temp = alist[leftmark]
                alist[leftmark] = alist[rightmark]
                alist[rightmark] = temp

        temp = alist[first]
        alist[first] = alist[rightmark]
        alist[rightmark] = temp

        return rightmark

    """
    build a balanced BST of the stocks based on the symbol. 
    Store the root of the BST as attribute bst, which is a TreeNode type.
    """

    def buildBST(self):
        #start_time = time.time()
        tree = AvlTree()
        for i in self.stockList: # scans through list
            TreeNode(i.symbol, i.val) # creates node for each symbol and value
            tree.put(i.symbol, i) # inserts each symbol and value into the avl tree
        self.bst = tree.root # sets bst to root of the avl tree
        #print(str(time.time() - start_time)+" seconds")
        return tree
        pass

    """
    Search a stock based on the symbol attribute. 
    It returns the details of the stock as described in __str__() function 
    or a “Stock not found” message when there is no match. 
    """

    def searchBST(self, query, current='dnode'):
        tree = self.buildBST() # calls the build bst
        finder = tree.get(query) # gets the input value from the tree
        if finder == None:
            return "Stock not found" # if not found
        else:
            return finder # returns the value found

    #def searchBST_test(self):  # searching bst test function for task 8
        #alist = []
        #for i in range(100):
            #n = random.randint(1, 1112)
            #alist.append(n)

        #start_time = time.time()
        #for i in alist:
            #self.searchBST(self.stockList[i].symbol)
        #print(str(time.time() - start_time) + " seconds")

    #def Linearsearch(self):
        #alist = []
        #for i in range(100):
            #n = random.randint(1, 1112)
            #alist.append(n)

        #start_time = time.time()
        #for i in alist:
            #for j in self.stockList:
                #if self.stockList[i].symbol == self.stockList[j].symbol:
                    #return i
        #print(str(time.time() - start_time) + " seconds")

    def longest_name(self): # finds the longest name in the list of stocks
        largest = self.stockList[0].sname # initializes the largest name to the first name
        largest_i = self.stockList[0] # initializes the stock object with to the first object
        for i in self.stockList[1:]: # starts at index 1
            temp = i.sname # name of the comparison value
            temp_i = i # stock object of the comparison value
            if len(largest) >= len(temp):
                largest = largest # if largest in indeed the largest nothing changes
                largest_i = largest_i
            else: # if not then temporary value is set to largest
                largest = temp
                largest_i = temp_i

        index = self.stockList.index(largest_i) # finds the index of the stock with the largest name
        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] # sets x values
        y = self.stockList[index].prices[3:] # sets y values
        print(self.stockList[index].sname)
        plt.plot(x, y) # plots values
        plt.title(str(largest))
        plt.xlabel("x")
        plt.ylabel("Prices $")
        plt.show()
        return largest, largest_i

    def largest_percent(self): # finds largest percent change
        start = float(self.stockList[0].prices[0]) # price of first stock at beginning
        end = float(self.stockList[0].prices[-1])  # prices of the last stock at the end
        percent_change = ((end - start) / start)*100 # percent change formula

        largest = percent_change # initializes largest as percent change of the first stock
        largest_info = self.stockList[0] # initializes info of the largest stock as first stock
        for k in self.stockList[1:]: # starts at index 1
            temp_start = k.prices[0] # first price
            temp_end = k.prices[-1] # final price
            temp = ((float(temp_end) - float(temp_start)) / float(temp_start))*100 # percent change formula
            temp_info = k
            if largest >= temp: # if larger nothing changes
                largest = largest
                largest_info = largest_info
            else: # if not then temporary is set to largest
                largest = temp
                largest_info = temp_info
        return str(largest_info)+": "+str(largest)+"%"

    def smallest_percent(self): # finds largest percent change
        start = float(self.stockList[0].prices[0]) # price of first stock at beginning
        end = float(self.stockList[0].prices[-1])  # prices of the last stock at the end
        percent_change = ((end - start) / start)*100 # percent change formula

        smallest = percent_change # initializes smallest as percent change of the first stock
        smallest_info = self.stockList[0] # initializes info of the smallest stock as first stock
        for k in self.stockList[1:]: # starts at index 1
            temp_start = k.prices[0] # first price
            temp_end = k.prices[-1] # final price
            temp = ((float(temp_end) - float(temp_start)) / float(temp_start))*100 # percent change formula
            temp_info = k
            if smallest <= temp: # if smaller nothing changes
                smallest = smallest
                smallest_info = smallest_info
            else: # if not then temporary is set to smallest
                smallest = temp
                smallest_info = temp_info
        return str(smallest_info)+": "+str(smallest)+"%"


# WRITE YOUR OWN TEST UNDER THIS IF YOU NEED
if __name__ == '__main__':

    stockLib = StockLibrary()
    testSymbol = 'GE'
    testName = 'General Electric Company'

    print("\n-------load dataset-------")
    stockLib.loadData("stock_database.csv")
    print(stockLib.size)

    print("\n-------linear search-------")
    print(stockLib.linearSearch(testSymbol, "symbol"))
    print(stockLib.linearSearch(testName, "name"))

    print("\n-------quick sort-------")
    print(stockLib.isSorted)
    stockLib.quickSort()
    print(stockLib.isSorted)

    print("\n-------build BST-------")
    stockLib.buildBST()

    print("\n---------search BST---------")
    print(stockLib.searchBST(testSymbol))

    print("\n---------longest---------")
    stockLib.longest_name()

    print("\n---------largest percent---------")
    print(stockLib.largest_percent())

    print("\n---------small percent---------")
    print(stockLib.smallest_percent())

    #print("\n---------Search BST test---------")
    #print(stockLib.searchBST_test())

    #print("\n---------Linear search test test---------")
    #print(stockLib.Linearsearch())



