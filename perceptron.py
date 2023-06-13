import numpy as np
import csv


def readdata(file):
    data = []
    with open(file) as f:
        reader = csv.reader(f)
        while True:
            try:
                line = next(reader)
                data.append(list(line))
            except StopIteration:
                break

    perceptronvector = []

    for line in data:
        perceptronvector.append([np.array(line[:-1]).astype(float), line[-1]])

    return perceptronvector


trainingdata = readdata("perceptron.data")

testdata = readdata("perceptron.test.data")

weightvector = np.random.rand(trainingdata[0][0].__len__())

print(weightvector)
print(trainingdata)

theta = 0
trainingaccuracy = 0
epo = 0

alpha = float(input("Enter the alpha factor \n"))

wantedaccuracy = float(input("Enter the accuracy you want to achieve on training dataset \n"))

# 0 - IRIS-VIRGINICA
# 1 - IRIS-VERSICOLOR

while trainingaccuracy < wantedaccuracy:

    correctguess = 0

    for k in trainingdata:
        if np.dot(weightvector, k[0]) - theta >= 0:
            print(f"{k[1]} labeled as 1 (Iris-versicolor)")
            if k[1] == "Iris-versicolor":
                correctguess = correctguess + 1
            else:
                print("Incorrect guess!")
                weightvector = weightvector + (alpha * (0 - 1) * k[0])
                theta = theta - (alpha * (0 - 1))
        else:
            print(f"{k[1]} labeled as 0 (Iris-virginica)")
            if k[1] == "Iris-virginica":
                correctguess = correctguess + 1
            else:
                print("Incorrect guess!")
                weightvector = weightvector + (alpha * (1 - 0) * k[0])
                theta = theta - (alpha * (1 - 0))
        epo = epo + 1
    trainingaccuracy = correctguess / trainingdata.__len__()
    print(f"Number of correct guesses {correctguess} / {trainingdata.__len__()}")

print(f"Accuracy of the testing model {trainingaccuracy}")
print(f"The amount of epochs it took is {epo}")

print("TESTING DATA NOW")

testacurracy = 0
correctlabel = 0

for i in testdata:
    if np.dot(weightvector, i[0]) - theta >= 0:
        if i[1] == "Iris-versicolor":
            correctlabel = correctlabel + 1
        else:
            print(f"Incorrect labeling: {i[1]} labeled as 1 (Iris-versicolor)")
    else:
        if i[1] == "Iris-virginica":
            correctlabel = correctlabel + 1
        else:
            print(f"Incorrect labeling: {i[1]} labeled as 0 (Iris-virginica)")

testacurracy = correctlabel / testdata.__len__()

print(f"Number of correct guesses {correctlabel} / {testdata.__len__()}")
print(f"Accuracy of the model {testacurracy}")

userinput = input("Press enter to add new vector, if you want to quit type quit \n")

while userinput != "quit":
    vector_components = []
    while len(vector_components) < testdata[0][0].__len__():
        component_str = float(input(f"Enter component {len(vector_components) + 1}: "))

        vector_components.append(component_str)

    vector = np.array(vector_components)

    print("Vector: ", vector)
    if np.dot(weightvector, vector) - theta >= 0:
        print("Predicted label is Iris-versicolor")
    else:
        print("Predicted label is Iris-virginica")

    userinput = input("Enter enter to add new vector, if you want to quit type quit \n")
