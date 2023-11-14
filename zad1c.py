user_in = input("Enter list: ")
input_list = [int(x) for x in user_in.split()]
output_list = [x**2 for x in input_list if x > 0]

print("Kwadraty wprowadzonych dodatnich liczb: ", output_list)
