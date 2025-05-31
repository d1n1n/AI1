import matplotlib.pyplot as plt
import numpy as np

# Task 1: Function plot
x = np.linspace(-10, 10, 500)
y = x ** 2

plt.plot(x, y)
plt.title("Graph of y = x²")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()



# Task 2: Histogram
data = np.random.normal(5, 2, 1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Normal Distribution (mean=5, std=2)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



# Task 3: Pie chart
labels = ['Coding', 'Music', 'Sports', 'Travel']
sizes = [30, 20, 25, 25]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("My Hobbies")
plt.axis('equal')  
plt.show()



# Task 4: Box plot for fruit weights
apple = np.random.normal(150, 10, 100)
banana = np.random.normal(120, 15, 100)
orange = np.random.normal(130, 12, 100)
pear = np.random.normal(140, 8, 100)

data = [apple, banana, orange, pear]

plt.boxplot(data, labels=['Apples', 'Bananas', 'Oranges', 'Pears'])
plt.title("Box Plot of Fruit Weights (100 each)")
plt.xlabel("Fruit")
plt.ylabel("Weight (grams)")
plt.grid(True)
plt.show()
