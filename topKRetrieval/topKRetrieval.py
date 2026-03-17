import heapq
from operator import itemgetter

def topKRetrieval(inputPose, otherPoses, distanceFunction, k):
    """
    Retrieves closest k poses to inputPose
    
    :param inputPose: User pose data
    :param otherPoses: Dictionary of pose dataset
    :param distanceFunction: Metric to calculate distance between poses
    :param k: Number of desired poses to ouput
    """

    poseDistanceDict = {}

    #Dictionary mapping poses in otherPoses with relative distance to inputPose
    for otherPose in otherPoses:
        relDist = distanceFunction(inputPose, otherPose)
        poseDistanceDict[otherPose] = relDist
        
    #Heapify and return closest K poses
    topK = heapq.nsmallest(k, poseDistanceDict.items(), key = itemgetter(1))

    return topK
