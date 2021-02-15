import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.

        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.

        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.

        num_entries = self.chipo.shape[0]
        return num_entries
    
    def info(self) -> None:
        # TODO
        # print data info.

        print (self.chipo.info(verbose=False))
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset

        num_col = self.chipo.shape[1]
        return num_col
        
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.

        for col in self.chipo.columns:
            print(col)
    
    def most_ordered_item(self):
        # TODO

        new_df = self.chipo.groupby(["item_name"])["quantity", "order_id"].sum().sort_values(by ='quantity', ascending = False)
        name = new_df.index.values[0]
        order_id = new_df["order_id"].iloc[0]
        quantity = new_df["quantity"].iloc[0]
        return name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?

       total_num = self.chipo["quantity"].sum()
       return total_num
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        self.chipo["item_price"]= self.chipo["item_price"].astype(str).apply(lambda x:  x.lstrip('$')).astype(float)
        df = self.chipo["item_price"] * self.chipo["quantity"]
        total = df.sum()
        return total
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?

        num = self.chipo["order_id"].nunique()
        return num
    
    def average_sales_amount_per_order(self) -> float:
        # TODO

        total_sale = self.total_sales()
        num_order = self.num_orders()
        ave = format(total_sale/num_order, '.2f')
        return float(ave)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?

        num = self.chipo["item_name"].nunique()
        return num
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        #print (letter_counter)
        df = pd.DataFrame.from_dict(letter_counter, orient='index', columns=['count'])
        df_sort = df.sort_values(by='count', ascending=False).head(x)
        item = list(df_sort.index.values)
        num = list(df_sort['count'])
        plt.bar(item, num, width = 0.4) 
        plt.xlabel("Items") 
        plt.ylabel("Number of Orders") 
        plt.title("Most popular items") 
        plt.show() 
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        x = list(self.chipo.groupby(["order_id"])["item_price"].sum())
        y = list(self.chipo.groupby(["order_id"])["quantity"].sum())
        color = "blue"
        plt.scatter(x, y, s=50, c=color)
        plt.xlabel("Order Price") 
        plt.ylabel("Num Items") 
        plt.title("Numer of items per order price") 
        plt.show()       

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    #assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()

    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    