import math
import pandas as pd
import random
import time

# Building block functions:
    
def deepcopy(clist):
    returnlist = []
    for partlist in clist:
        returnlist.append(partlist.copy())
    return returnlist

def dist(x1, x2):
    """ Return euclidean distance between x1 and x2 """
    dist_sqr = 0
    for i in range(len(x1)):
        dist_sqr += (x1[i]-x2[i])**2
    dist = math.sqrt(dist_sqr)
    return dist

def centroid(xList):
    """ Compute centroid of multi-dimensional x data in xList """
    center = xList.sum()/len(xList)
    return center.tolist()

def assignmentDiffers(yCurrent, yPrev):
    """ Return True if yCurrent differs from yPrev """

    for i in range(len(yCurrent)):
        # make sure they're the same length
        if len(yCurrent[i])!=len(yPrev[i]):
            print("length difference: "+str(i))
            return True
        # check if each item is equal
        for j in range(len(yCurrent[i])):
            if yCurrent[i][j]!=yPrev[i][j]:
                print("value difference\ncurrent"+str(yCurrent[i][j])+"\nprevious\n"+str(yPrev[i][j]))
                return True
    return False

# Let's use a class for our K-Means implementation
class KMeans:
    """ Perform k-means clustering """
    
    def __init__(self, k=5):
        self.k = k          # number of clusters
        self.means = []  # means of clusters
        self.MAX_ITTERATIONS = 50
        
    def classify(self, x):
        """ Return the index of the cluster closest to the input """
        smallest_dist = None
        smallest_index = 0
        for i in range(len(self.means)):
            current_dist = dist(x,self.means[i])
            if smallest_dist==None:
                smallest_dist = current_dist
                continue
            if smallest_dist > current_dist:
                smallest_dist = current_dist
                smallest_index = i
        #print(str(smallest_index)+" "+str(smallest_dist))
        return smallest_index
    
    def train(self, data):
        """ Train model based on data """
        # Initial random cluster assignment
        initial_clusters = random.sample(range(len(data)), self.k)
        #print(initial_clusters)
        for x in range(self.k):
            #print(data.iloc[initial_clusters[x]].tolist())
            self.means.append(data.iloc[initial_clusters[x]].tolist())
        print(self.means)
        
        # place the data's index into a 2d array.  each 1st dimention index is a cluster
        clusters = [[] for ind in range(self.k)]
        new_clusters = [[] for ind in range(self.k)]
        assignChange=True
        itterations = 0
        while assignChange and itterations<=self.MAX_ITTERATIONS:
            itterations+=1
            print(itterations)
            #print("previous\n"+str(clusters))
            #print("current\n"+str(new_clusters))
            
            # preform training
            for index in range(len(data)):
                classification = self.classify(data.iloc[index])
                #print(classification)
                #time.sleep(.2)
                new_clusters[classification].append(index)
                
            
            #print("new\n"+str(new_clusters))
            #print("old\n"+str(clusters))
            
            # check if any changes occured in the clustering
            assignChange=assignmentDiffers(new_clusters,clusters)
            # update the cluster centroids to reflect the updated clusters
            if assignChange:
                for i in range(self.k):
                    self.means[i] = centroid(data.iloc[new_clusters[i]])
                print(self.means)
                clusters = deepcopy(new_clusters)
                new_clusters = [[] for ind in range(self.k)]
        
        

class DBSCAN:
    """ Perform DBSCAN clustering """
    def __init__(self,epsilon,minPts):
        self.epsilon = epsilon
        self.minPts = minPts
    
    def classify(self,x):
        """ Return the index of the cluster to which this point would belong, or None """

        pass
    


    def train(self,data):
        """ Train model based on data """

        corePoints = [] # Contains True or False depending if the corresponding point is a core point
        for pt in data:
            # count points in radius from pt
            closePts = 0
            for pt2 in data:
                if pt == pt2:
                    continue
                dis = dist(pt,pt2)
                if dis <= self.epsilon:
                    closePts += 1
            if closePts >= self.minPts:
                corePoints.append(True)
            else:
                corePoints.append(False)
    
        clusterAssignments = [None]*len(data)
        nextClusterIndex = 0
        while tmp:

            unassignedCorePointIndices = []
            for pt1Index in len(data):
                pt
            todoList= []
            startingPointIndex = 0
            todoList.append(startingPointIndex)
            clusterAssignments[startingPointIndex] = nextClusterIndex

            while len(todoList) > 0:
                for pt1 in todoList:
                    for pt2Index in len(data):
                        pt2 = data[pt2Index]
                        if pt1==pt2:
                            continue
                        d=dist(pt1,pt2)
                        if d <= self.epsilon:
                            if corePoints[pt2Index] == True:
                                todoList.append(pt2Index)
                                clusterAssignments[pt2Index] = nextClusterIndex
            
            for pt1Index in len(data):
                pt1 = data[pt1Index]
                if clusterAssignments[pt1Index] == nextClusterIndex:
                    for pt2Index in len(data):
                        pt2 = data[pt2Index]
                        if pt1==pt2:
                            continue
                        if clusterAssignments[pt2Index] == None and corePoints[pt2Index] == False:
                            if dist(pt1,pt2) <= self.epsilon:
                                clusterAssignments[pt2Index] = nextClusterIndex

            for pt1Index in len(data):
                if corePoints[pt1Index] == True and clusterAssignments[pt1Index] == None:
                    anyCorePointsRemainingUnassigned = True
            if anyCorePointsRemainingUnassigned == False:
                pass

    def train2(self,data):
        """ Train model based on data """

        corePoints = [] # the index of all core points
        nonCorePoints = [] # the index of all non-core points
        outliers = [] # the index of all points with no neighbors

        data["cluster"] = -1 # setting the default cluster of every point to an "unassigned" value

        for pt in data:
            # count points in radius from pt
            closePts = 0
            for pt2 in data:
                if pt == pt2:
                    continue
                dis = dist(pt,pt2)
                if dis <= self.epsilon:
                    closePts += 1
            if closePts >= self.minPts:
                corePoints.append(pt.name)
            elif closePts >0:
                nonCorePoints.append(pt.name)
            else:
                outliers.append(pt.name)
        
        # assigning the core points to clusters 
        # the first cluster isn't random because I don't see a value in it being random.
        # to change it to use a random, change the function below to get a list of every unassigned core point then return a random value from that
        cluster=0
        pt = findFirstUnassigned(data[corePoints])
        while pt!=None:

            # List of every point added to the cluster to make sure every core point around them gets added as well
            currentCluster = [pt]  # list init
            data[pt]["cluster"] = cluster  
            currentClusterIndex=0  # current place in the cluster

            # loop until every point that has been added has been used
            while currentClusterIndex<len(currentCluster):
                for pt2Index in corePoints:
                    if dist(data[pt],data[pt2Index])<=self.epsilon:
                        # checks if no cluster has been assigned to the point
                        if data[pt2Index]["cluster"] == -1:
                            currentCluster.append(pt2Index)  #adds it to the list
                            data[pt2Index]["cluster"] = cluster
                # itterates to the next point in the cluster
                currentClusterIndex += 1
                # error check
                if currentClusterIndex<len(currentCluster):
                    pt = currentCluster[currentClusterIndex]
                else:
                    break
            # updates to the next cluster name and resets the starting point
            cluster+=1
            pt = findFirstUnassigned(data[corePoints])
        
        # assigning the edge points to clusters
        for pt in nonCorePoints:
            # in case of multiple (but not more than min) core points being close to the edge point
            # in the case of a tie it's up to whatever the max function makes it
            clusters = {}
            for corePtIndex in corePoints:
                if dist(data[pt],data[corePtIndex]) <= self.epsilon:
                    try:
                        clusters[data[corePtIndex]["cluster"]]+=1
                    except e:
                        clusters[data[corePtIndex]["cluster"]] = 1
            # sets the cluster to the max value 
            # !!potential for bugs here!!
            data[pt]["cluster"] = max(clusters,key=clusters.get)

            
            


    def findFirstUnassigned(self, data):
        for dat in data:
            if dat["cluster"]==-1:
                return dat.name  # return the index of the first instance of a point without a cluster assignment
        return None

def kmeans(x, k):
    # Use it like this?
    km = KMeans(k)
    km.train(x[x.columns[:-1]])
    # can print out km.means to see the fit means
    # can call km.classify([1,2,3,4]) to get cluster index
    
    #The function should return a list the length of x that contains
    # the cluster number (1 - k) for the corresponding x point
    # TODO determine return value

def main():
    path=""
    infile="clustering_dataset_01.csv"
    df = pd.read_csv(path+infile,header=None)

    kmeans(df,5)

if __name__=="__main__":
    main()