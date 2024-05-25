import numpy as np
# Do not import any other libraries.


class KMeans:

    def __init__(self, X, n_clusters):
        self.X = X
        self.n_clusters = n_clusters
        self.centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    def mean(self, value):
        """
        Calculate mean of the dataset column-wise.
        Do not use built-in functions

        :param value: data
        :return the mean value
        """
        #Number of data points
        n = len(value)
        #Initilazing a list to hold the sum of each column
        sum_values = [0] * len(value[0])

        #Sum each column
        for row in value:
            for i in range(len(row)):
                sum_values[i] += row[i]

        #Calculating mean_value for each column
        mean_values = [sum_val / n for sum_val in sum_values]

        return mean_values

    def std(self):
        """
        Calculate standard deviation of the dataset.
        Use the mean function you wrote. Do not use built-in functions

        :param X: dataset
        :return the standard deviation value
        """
        #Calculate the mean
        means = self.mean(self.X)

        #Number of data points
        n = len(self.X)

        #Initialize a list to hold the sum of squared deviations for each column
        sum_squared_devs = [0] * len(self.X[0])

        #Calculate the squared deviations from the mean for each column
        for row in self.X:
            for i in range(len(row)):
                sum_squared_devs[i] += (row[i] - means[i]) ** 2

        #Calculate the variance
        variances = [sum_squared_dev / n for sum_squared_dev in sum_squared_devs]

        #Calculate the standard deviation
        std_devs = [variance ** 0.5 for variance in variances]

        return std_devs

    def standard_scaler(self):
        """
        Implement Standard Scaler to X.
        Use the mean and std functions you wrote. Do not use built-in functions

        :param X: dataset
        :return X_scaled: standard scaled X
        """

        #Calculating mean and std
        means = self.mean(self.X)
        std_deviation = self.std()

        #Calculating standart scaler
        X_scaled = []
        for row in self.X:
            scaled_row = [(val - mean) / std if std != 0 else 0 for val, mean, std in zip(row, means, std_deviation)]
            X_scaled.append(scaled_row)

        return np.array(X_scaled)

    def euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two data points.
        Do not use any external libraries

        :param point1: data point 1, list of floats
        :param point2: data point 2, list of floats

        :return the Euclidean distance between two data points
        """

        #Intializing the sum of squared differences
        sum_squared_diff = 0

        #Iterate throught each dimension and calculate the squared difference
        for i in range(len(point1)):
            sum_squared_diff += (point1[i] - point2[i])**2

        #Return the square root of value
        return sum_squared_diff ** .5

    def get_closest_centroid(self, point):
        """
        Find the closest centroid given a data point.

        :param point: list of floats
        :param centroids: a list of list where each row represents the point of each centroid
        :return: the number(index) of the closest centroid
        """

        #Initalizing the minimum distance to a large value
        min_dist = float("inf")
        #Will return this value
        closest_centroid_index = -1

        #Iterating through each centroid
        for i, centroid in enumerate(self.centroids):
            #Calculate Euclidean distance
            distance = self.euclidean_distance(point, centroid)

            #Update the closest_centroid_index
            if distance <= min_dist:
                min_dist = distance
                closest_centroid_index = i

        return closest_centroid_index

    def update_clusters(self):
        """
        Assign all data points to the closest centroids.
        Use "get_closest_centroid" function

        :return: cluster_dict: a  dictionary  whose keys are the centroids' key names and values are lists of points that belong to the centroid
        Example:
        list_of_points = [[1.1, 1], [4, 2], [0, 0]]
        centroids = [[1, 1],
                    [2, 2]]

            print(update_clusters())
        Output:
            {'0': [[1.1, 1], [0, 0]],
             '1': [[4, 2]]}
        """
        #Initalizing a cluster dictionary
        c_dict = {str(i) : [] for i in range(len(self.centroids))}

        #Assigning each point to the closest centroid
        for point in self.X:
            closest_centroid_index = self.get_closest_centroid(point)
            c_dict[str(closest_centroid_index)].append(point)

        return c_dict


    def update_centroids(self, cluster_dict):
        """
        Update centroids using the mean of the given points in the cluster.
        Doesn't return anything, only change self.centroids
        Use your mean function.
        Consider the case when one cluster doesn't have any point in it !

        :param cluster_dict: a  dictionary  whose keys are the centroids' key names and values are lists of points that belong to the centroid

        """
        #Initializing a list
        updated_centroid = []

        for key in cluster_dict:
            points = cluster_dict[key]
            if len(points) > 0:
                cluster_mean = self.mean(points)
                updated_centroid.append(cluster_mean)
            else:
                #If cluster empty keep the previous centroid
                updated_centroid.append(self.centroids[int(key)])

        #Update centroids
        self.centroids = np.array(updated_centroid)

    def converged(self, clusters, old_clusters):
        """
        Check the clusters converged or not

        :param clusters: new clusters, dictionary where keys are cluster labels and values are the points(list of list)
        :param old_clusters: old clusters, dictionary where keys are cluster labels and values are the points(list of list)
        :return: boolean value: True if clusters don't change
        Example:
        clusters = {'0': [[1.1, 1], [0, 0]],
                    '1': [[4, 2]]}
        old_clusters = {'0': [[1.1, 1], [0, 0]],
                        '1': [[4, 2]]}
            print(update_assignment(clusters, old_clusters))
        Output:
            True
        """
        #Check if the number of clusters is different
        for key in clusters:
            if not np.array_equal(clusters[key], old_clusters[key]):
                return False
        return True
    def calculate_wcss(self, clusters):
        """

        :param clusters: dictionary where keys are clusters labels and values the data points belong to that cluster
        :return:
        """
        wcss = 0

        for key in clusters:
            #Center of cluster
            centroid = self.centroids[int(key)]
            #Points in cluster
            cluster_points = clusters[key]

            for point in cluster_points:
                #Euclidean distance
                distance = self.euclidean_distance(point, centroid)
                #Add square of distance
                wcss += distance ** 2

        return wcss

    def fit(self):
        """
        Implement K-Means clustering until the clusters don't change.
        Use the functions you have already implemented.
        Print how many steps does it take to converge.
        :return: final_clusters: a  dictionary  whose keys are the centroids' key names and values are lists of points that belong to the centroid
                 final_centroids: list of list with shape (n_cluster, X.shape[1])
                 wcss: within-cluster sum of squares
        """
        converged = False
        steps = 0
        #
        while not converged:
            #Old clustering results
            old_clusters = self.update_clusters()
            #Update centroids
            self.update_centroids(old_clusters)
            #New clustering results
            final_clusters = self.update_clusters()
            #Check for convergence
            converged = self.converged(final_clusters, old_clusters)
            steps += 1

        wcss = self.calculate_wcss(final_clusters)
        final_centroids = self.centroids

        print("Number of steps: {}".format(steps))
        return final_clusters, final_centroids, wcss
