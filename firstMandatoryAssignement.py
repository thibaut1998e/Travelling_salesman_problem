import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import time
import random as rd



def loadData():
    file = open("european_cities.csv")
    donnees = file.read()
    donnees = donnees.split("\n")

    for i in range(len(donnees)):
        donnees[i] = donnees[i].split(";")
    villes = donnees[0]

    donnees = donnees[1:len(donnees)-1]
    for i in range(len(donnees)):
        for j in range(len(donnees[0])):

            donnees[i][j] = float(donnees[i][j])

    return donnees, villes


matriceDistances, villes = loadData()

def transformNumerWalkIntocityWalk(numberWalk):
    return [villes[index] for index in numberWalk]



def BruteForceTSP(currentCity, bestDistance, CurrentDistance, bestWalk, currentWalk, citiesToVisit):
    if len(citiesToVisit)==0:
        if CurrentDistance + matriceDistances[currentCity][currentWalk[0]] < bestDistance:
            bestDistance = CurrentDistance + matriceDistances[currentCity][currentWalk[0]]
            bestWalk = currentWalk

    else:
        for city in citiesToVisit:
            if CurrentDistance < bestDistance:
                bestWalk, bestDistance = BruteForceTSP(city, bestDistance, CurrentDistance + matriceDistances[currentCity][city],
                          bestWalk, currentWalk+[city], list(set(citiesToVisit)-set([city])))
    return bestWalk, bestDistance


def exactTsp(listOfCities):
    bestWalk, bestDistance = BruteForceTSP(listOfCities[0], 100000, 0, [], [listOfCities[0]], listOfCities[1:])
    return bestWalk, bestDistance



def evaluateTimeOfBruteForce(nbCity):
    t1 = time.time()
    exactTsp(range(nbCity))
    t2 = time.time()
    return t2-t1

#plot the time resolution depending on the number of cities.
def plotTimesGraph(nbMaxCities):
    X = range(1, nbMaxCities+1)
    listTime = [evaluateTimeOfBruteForce(i) for i in X]
    plt.plot(X, listTime)
    plt.xlabel("numberOfCities")
    plt.ylabel("time of resolution in seconds")
    plt.show()

#Q1 : time of resolution exhaustive search
#plotTimesGraph()
#solution : 11 s for 12 cities


def randomSolution(listCities):
    return list(np.random.permutation(listCities))



def evaluateSolution(solution):
    distance = 0
    for i in range(len(solution)-1):
        distance += matriceDistances[solution[i]][solution[i+1]]
    return distance + matriceDistances[solution[-1]][solution[0]]


"""swap the city i and j in the permutation listOfCities"""
def swap(listOfCities, i, j):
    newList = copy(listOfCities)
    newList[i], newList[j] = newList[j], newList[i]
    return newList




"""The neighboorhood is defined as all the permutation obtained by one swap of two cities
At each step we take the best neighboor of the current solution and set it as the current solution
We stop when we can't find a better solution in the neighboorhood of the current solution
The function doesnt modify initialSolution"""
def localSearch(initialSolution, nbMaxIteration):
    bestValue = 1000000
    bestWalk = copy(initialSolution)
    stop = False
    cpt = 0
    while not stop and cpt < nbMaxIteration:
        solution = copy(bestWalk)

        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                potentialSolution = swap(solution, i, j)
                value = evaluateSolution(potentialSolution)
                if value < bestValue:
                    bestValue = value
                    bestWalk = copy(potentialSolution)
        stop = (solution == bestWalk)

        cpt += 1

    return bestWalk, bestValue

def copy(list):
    newList = []
    for i in range(len(list)):
        newList.append(list[i])
    return newList




"""select randomly 20 differents initial solutions and performs hillClimbing on them"""
def hillClimbing(listCities):
    listSolutions = []
    for i in range(20):
        solution = randomSolution(listCities)
        bestWalk, bestValue = localSearch(solution, 100000)
        listSolutions.append(bestWalk)
        listSolutions.sort(key=evaluateSolution)
        bestSolution = listSolutions[0]

    return bestSolution, listSolutions




def statistics(listSolutions):
    valuesOfSolution = [evaluateSolution(sol) for sol in listSolutions]
    return  np.mean(valuesOfSolution), \
           max(valuesOfSolution), min(valuesOfSolution), np.sqrt(np.var(valuesOfSolution))



"""plot the ration between the exact solution and the mean, best and worst solution found depending on the number of cities"""
def plotStatistics(nbMaxCities):
    listMean = []
    listMax = []
    listMin = []
    listStandDev = []
    for nbCities in range(2,nbMaxCities+1):
        bestValue = exactTsp(range(nbCities))[1]
        mean, maximum, minimum, standDev = statistics(hillClimbing(range(nbCities))[1])
        listMean.append(bestValue/mean)
        listMax.append(bestValue/maximum)
        listMin.append(bestValue/minimum)
        listStandDev.append(standDev)

    X = range(2, nbMaxCities+1)

    plt.plot(X, listMin)
    plt.plot(X, listMax)
    plt.plot(X, listMean)
    plt.xlabel("numberOfCities")
    plt.title("ratios between the best score and the mean, max and min score")
    plt.show()





"""Genetic Algorithm"""

def geneticAlgorithm(mutationRate, popSize, nbIteration, listCities, n, type="C"):

    population = generateInitialPopulation(popSize, listCities, type)
    fitnessThroughGenerations = []
    for i in range(nbIteration):
        selectedParents = selectionParents(population)
        reproductionParents(population, selectedParents, mutationRate, type)
        population.sort(key=getValueOf)
        fitnessThroughGenerations.append(getValueOf(population[0]))
        population = population[:n] + rd.sample(population[n + 1:], popSize - n)

    return population, fitnessThroughGenerations




def generateInitialPopulation(size, listCities, type):
    initialPopulation = []
    if type == "C" or type == "L":
        sol = randomSolution(listCities)
        for i in range(size):
            initialPopulation.append((sol, evaluateSolution(sol)))

    if type == "B":
        for i in range(size):
            sol = randomSolution(listCities)

            initialPopulation.append((sol, localSearch(sol,100)[1]))

    return initialPopulation

def getPermutation(solution):
    return solution[0]

def getValueOf(solution):
    return solution[1]




def selectionParents(population):
    listScore = []
    sum = 0
    listCostFunction = [getValueOf(ind) for ind in population]
    costMax = max(listCostFunction)

    for i in range(len(population)):
        score = costMax + 1 - listCostFunction[i]
        listScore.append(score)
        sum += score
    listProba = []
    for i in range(len(listScore)):
        listProba.append(listScore[i]/sum)
    listSelectedParents = []

    for i in range(len(population)):

        selectedParent = population[np.random.choice(range(len(population)), p=listProba)]
        listSelectedParents.append(selectedParent)

    return listSelectedParents




def reproductionParents(population, listSelectedParents, mutationRate, type):

    for i in range(len(listSelectedParents)//2):
        son, daughter = partiallyMapCrossover(getPermutation(listSelectedParents[2*i]), getPermutation(listSelectedParents[2*i+1]))
        #son, daughter = cp.copy(getPermutation(listSelectedParents[2*i])), cp.copy(getPermutation(listSelectedParents[2*i+1]))
        mutation(son, mutationRate)
        mutation(daughter, mutationRate)
        if type == "C":
            population.append((son, evaluateSolution(son)))
            population.append((daughter, evaluateSolution(daughter)))

        if type == "L" or type == "B":
            locSearchSon, sonLocSearchValue = localSearch(son, 20)
            localSearchDaughter, daughterLocSearchValue = localSearch(daughter, 20)
            if type == "L":
                population.append((locSearchSon, sonLocSearchValue))
                population.append((localSearchDaughter, daughterLocSearchValue))
            if type == "B":
                population.append((son, sonLocSearchValue))
                population.append((daughter, daughterLocSearchValue))



def partiallyMapCrossover(mum, dad):
    def onePregnancy(mainParent, otherParent):

        dictOtherParent = dict(zip(otherParent, range(len(otherParent))))
        daughter = [-1] * len(mainParent)
        randomNumber = rd.randint(0,len(mainParent) // 2)
        indexOfSelectedPart = range(randomNumber, randomNumber + len(mainParent) // 2, 1)
        selectedPart = mainParent[randomNumber:randomNumber + len(mainParent) // 2]
        daughter[randomNumber:randomNumber + len(mainParent) // 2] = selectedPart
        for i in indexOfSelectedPart:
            if otherParent[i] not in selectedPart:
                position = dictOtherParent[mainParent[i]]
                while position in indexOfSelectedPart:
                    position = dictOtherParent[mainParent[position]]
                daughter[position] = otherParent[i]
        for i in range(len(daughter)):
            if daughter[i] == -1:
                daughter[i] = otherParent[i]
        return daughter

    daughter = onePregnancy(mum, dad)
    son = onePregnancy(dad, mum)

    return daughter, son



def mutation(ind,mutationRate):
    for k in range(1):
        if rd.random() < mutationRate:
            i = rd.randint(0,len(ind)-1)
            j = rd.randint(0, len(ind)-1)
            while i == j:
                j = rd.randint(0, len(ind)-1)
            ind[i], ind[j] = ind[j], ind[i]






def plotBestSolutionFunctionOfN(mutationRate, nbGeneration, listOfCities, listPopSize):

    for popSize in listPopSize:
        X = range(1, popSize+1)
        Y = [np.mean([evaluateSolution(geneticAlgorithm(mutationRate, popSize, nbGeneration, listOfCities, n)[0][0][0]) for i in range(20)])
            for n in X]
        plt.plot(X,Y)
    plt.xlabel("Number of individuals selected non-ranomly in the Survivor's Selection phase")
    plt.title("best Solution found over " + str(nbGeneration) + " generation with a mutation rate = "+str(mutationRate) +
                " for the "+str(len(listOfCities))+" first cities for three differents population sizes")
    plt.xticks(np.arange(0, popSize, 1))
    plt.show()

def plotBestSolutionFunctionOfMutationRate(nbGeneration, listOfCities, listPopSize):

    for popSize in listPopSize:
        X = np.arange(0,1.1,0.1)
        Y = [np.mean(
            [evaluateSolution(geneticAlgorithm(mutationRate, popSize, nbGeneration, listOfCities, popSize)[0][0][0]) for i in
             range(20)])
             for mutationRate in X]
        plt.plot(X, Y)
    plt.xlabel("Mutation Rate")
    plt.title(
        "best Solution found over " + str(nbGeneration) + " genrations "
         + " for the " + str(len(listOfCities)) + " first cities for three differents population sizes")
    plt.xticks(np.arange(0, 1, 0.1))
    plt.show()





def computeStatistics(mutationRate, popSize, nbGeneration, listOfCities, n, type = "C"):
    scoresObtained = [getValueOf(geneticAlgorithm(mutationRate,popSize,nbGeneration,listOfCities,n, type)[0][0]) for i in range(20)]
    mean = np.mean(scoresObtained)
    standardDeviation = np.sqrt(np.var(scoresObtained))
    maxScore = max(scoresObtained)
    minScore = min(scoresObtained)
    return mean, standardDeviation, maxScore, minScore




def plotFitnessThroughGenerations(mutationRate, listPopSize, nbGenerations, listOfCities, listOfN, type = "C"):
    for j in range(3):

        
        popSize = listPopSize[j]
        n = listOfN[j]
        meanOfBestIndividual = np.zeros((1, nbGenerations))
        for i in range(20):
            fitnessThroughGenerations = geneticAlgorithm(mutationRate, popSize, nbGenerations, listOfCities, n, type)[1]
            meanOfBestIndividual += np.array([fitnessThroughGenerations])
        meanOfBestIndividual /= 20
        plt.plot(range(nbGenerations), list(meanOfBestIndividual[0]))
    plt.xlabel("number generations")
    plt.title("average fitness of best individuals through generations for three differnents population size")
    plt.show()



def answerToQuestions(question):
    #Q1 : exhaustive Search for the 6 firsts cities
    if question == "Q1":
        print("resolution for 10 cities")
        t1 = time.time()
        bestWalk, bestDistance = exactTsp(range(10))
        t2= time.time()
        print("best solution is : " + str(bestWalk) + " which corresponds to " + str(transformNumerWalkIntocityWalk(bestWalk)))
        print("best distance is : " + str(bestDistance))
        # solution : [0,1,9,4,5,2,6,8,3,7] =
        # ['Barcelona', 'Belgrade', 'Istanbul', 'Bucharest', 'Budapest', 'Berlin', 'Copenhagen', 'Hamburg', 'Brussels', 'Dublin'] distance = 7486.31
        print("time to solve TSP with 10 cities = "+ str(t2-t1))
        print("plot time function of the number of cities, it takes about 20s")
        plotTimesGraph(12)


    if question == "Q2":
        #Q2 hill climbing : find  permutations of the 24 city using local search from 20 initial solutions
        for numberCities in [10, 24]:
            print("RESULTS FOR " + str(numberCities) + " CITIES")
            if numberCities == 10:
                print("best value found by exhaustive search for 10 cities is " + str(exactTsp(range(10))[1]))
            bestSolution, listSolutions = hillClimbing(range(numberCities))
            print("best solution found over the 20 differents starting points : "+ str(bestSolution))
            print("which has the value : " + str(evaluateSolution(bestSolution)))
            print("it corresponds to the city walk : "+ str(transformNumerWalkIntocityWalk(bestSolution)))
            print("list of scores of all the solutions found from the 20 differents starting points : ")
            print([evaluateSolution(sol) for sol in listSolutions])
            mean, worst, best, standDev = statistics(listSolutions)
            print("mean : " + str(mean) + ", worst : " + str(worst) + ", best : " + str(best) + ", standard deviation : " + str(standDev))
        print("PLOT GRAPH THAT COMPARES PERFORMANCES OF BRUTEFORCE AND HILL CLIMBING")
        print("plot the ration between the exact solution and the mean, best and worst solution found depending on the number of cities")
        plotStatistics(12)


    if question == "Q3":
        print("TEST OF PARTIALLY MAP CROSSOVER")
        mum = [5,1,3,7,8,9,2,6,4,0]
        dad = [1,5,0,9,3,7,2,6,8,4]
        son, daughter = partiallyMapCrossover(mum, dad)
        print("the two parents " + str(mum) + " and " + str(dad))
        print("give children " + str(son) + " and " + str(daughter))
        print("\n")
        for numberCities in [10, 24]:
            print("RESULTS FOR " + str(numberCities) + " cities")
            if numberCities == 10:
                print("best value found by exhaustive search for 10 cities is " + str(exactTsp(range(10))[1]))
            print("RESULTS OF ONE RUN OF THE ALGORITHM")
            t1 = time.time()
            population = geneticAlgorithm(1, 20, 100, range(numberCities), 20, "C")[0]
            t2 = time.time()
            print("the whole population sorted at the end of the search : ")
            print([getPermutation(ind) for ind in population])
            print("corresponding values")
            print([getValueOf(ind) for ind in population])
            print("parameters used : popSize = 20, nbGenerations = 100, mutationRate = 1, n = 20")
            print("running time = " + str(t2-t1))
            print("\n")

        print("24 CITIES, COMPARAISON OF RESULTS FOR 3 DIFFERNENTS POPULATION SIZES")

        for popSize in [10,20,30]:
            print("POPULATION SIZE = " + str(popSize))
            mean, standDev, maxScore, minScore = computeStatistics(1,popSize,100,range(24),popSize)
            print("best solution has value : " + str(minScore) + ", worst : " + str(maxScore) + ", mean : "+str(mean) + ", strandard deviation " + str(standDev))
        print("parameters used : n = popsize, mutationrate =1, nbGénérations = 100")
        print("PLOT FITNESS OF BEST INDIVIDUAL THROUGH GENERATIONS 5AVERAGE OVER 20 RUNS")
        plotFitnessThroughGenerations(1, [10, 20, 30], 100, range(24), [10, 20, 30])
        print("PLOT BEST SOLUTION FUNCTION OF MUTATION RATE ")
        print("(it takes abaut 1min30)")
        plotBestSolutionFunctionOfMutationRate(100, range(24), [10, 20, 30])
        print("PLOT BEST SOLUTION FUNCTION OF N")
        print("(it takes about 3 minutes)")
        plotBestSolutionFunctionOfN(1, 100, range(24), [10, 20, 30])

    if question == "Q4":
        for type in ["L", "B"]:
           if type == "L":
               print("RESULTS OF LAMARCKIAN ALGORITHM")

           if type == "B":
               print("RESULTS OF BALDWINIAN ALGORITHM")

           for numberCities in [10,20]:
                print("RESULTS OF ONE RUN FOR " + str(numberCities) + " CITIES")
                if numberCities == 10:
                    print("best value found by exhaustive search for 10 cities is " + str(exactTsp(range(10))[1]))
                t1 = time.time()
                population = geneticAlgorithm(1, 10, 10, range(numberCities), 10, type)[0]
                t2 = time.time()
                print("the whole population sorted at the end of the search : ")
                print([getPermutation(ind) for ind in population])
                print("corresponding values")
                print([getValueOf(ind) for ind in population])
                print("parameters used : popSize = 10, nbGenerations = 10, mutationRate = 1, n = 10")
                print("running time = " + str(t2 - t1))

    if question == "Q4b":
        print("COMPUTE STATISTICS OF LAMARCKIAN AND BALDWINIAN")
        for type in ["L", "B"]:
            if type == "L":
                print("LAMARCKIAN")
            if type == "B":
                print("BALDWINIAN")

            for popSize in [6,10,14]:
                print("POPULATION SIZE = " + str(popSize))
                mean, standDev, maxScore, minScore = computeStatistics(1,popSize,10,range(20),popSize, type)
                print("best solution has value : " + str(minScore) + ", worst : " +
                      str(maxScore) + ", mean : "+str(mean) + ", strandard deviation " + str(standDev))

    if question == "Q4c":
        print("PLOT FITNESS OF BEST INDIVIDUAL THROUGH GENERATIONS AVERAGE OVER 20 RUNS LAMARCKIAN")
        plotFitnessThroughGenerations(1, [6, 10, 14], 10, range(24), [6, 10, 14], "L")
        print("PLOT FITNESS OF BEST INDIVIDUAL THROUGH GENERATIONS AVERAGE OVER 20 RUNS BALDWINIAN")
        plotFitnessThroughGenerations(1, [6, 10, 14], 10, range(24), [6, 10, 14], "B")


#answerToQuestions("Q1")
#answerToQuestions("Q2")
#answerToQuestions("Q3")
#nswerToQuestions("Q4")
#answerToQuestions("Q4b")
#answerToQuestions("Q4c")












