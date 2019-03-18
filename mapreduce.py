# standard way to square values 

def square(lst1):
    lst2 = []
    for num in lst1:
        lst2.append(num**2)
    return lst2
print(square([4,3,2,1]))

# using map and lambda
n = [4,3,2,1]
print(list(map(lambda x: x**2, n)))

# using list comprehension
print([x**2 for x in n])



