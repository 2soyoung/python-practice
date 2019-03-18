# https://www.youtube.com/watch?v=cKlnR-CB3tk
from functools import reduce
##################################
####### MAP #######################
# The standard way to square values 

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


#############################
########## FILTER ###########

# The standard way to filter items out of a sequence 
def over_two(lst1):
    lst2 = [x for x in lst1 if x>2]
    return lst2
print(over_two([4,3,2,1]))

# using filter and labmda
n  =  [4,3,2,1]
print(list(filter(lambda x: x>2, n )))

# using the list comprehension

# notice that this just returns a logical list of w
print([x > 2 for x in n])

# The correct usage of the list comprehension
print([x for x in n if x >2])

################################
########### REDUCE #############

# The standard way 
def mult(lst1):
    prod = lst1[0]
    for i in range(1, len(lst1)):
        prod *= lst1[i]
    return prod

print(mult([4,3,2,1]))

# using reduce and lambda 
n = [4,3,2,1]
print(reduce(lambda x, y: x*y, n))




