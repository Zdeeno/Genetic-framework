import re
import numpy as np

cities_set = []
cities_tups = []
cities_dict = {}


def read_tsp_data(tsp_name):
    tsp_name = tsp_name
    with open(tsp_name) as f:
        content = f.read().splitlines()
        cleaned = [x.lstrip() for x in content if x != ""]
        return cleaned


def detect_dimension(in_list):
    non_numeric = re.compile(r'[^\d]+')
    for element in in_list:
        if element.startswith("DIMENSION"):
            return non_numeric.sub("", element)


def get_cities(list, dimension):
    dimension = int(dimension)
    for item in list:
        for num in range(1, dimension + 1):
            if item.startswith(str(num)):
                index, space, rest = item.partition(' ')
                if rest not in cities_set:
                    cities_set.append(rest)
    return cities_set


def city_tup(list):
    for item in list:
        first_coord, space, second_coord = item.partition(' ')
        cities_tups.append((first_coord.strip(), second_coord.strip()))
    return cities_tups


def produce_final(file="../DU1/berlin52.tsp"):
    """
    get coordinates of all places
    :param file: name of .tsp file
    :return: numpy array of size [length, 2]
    """
    data = read_tsp_data(file)
    dimension = detect_dimension(data)
    cities_set = get_cities(data, dimension)
    cities_tups = city_tup(cities_set)
    return np.asarray(cities_tups, dtype=float)


def distance_matrix(file="../DU1/berlin52.tsp"):
    arr = produce_final(file)
    print(arr)
    print(arr.shape)
    print(arr[1, :])
    print(arr - arr[0, :])
    ret = np.zeros((arr.shape[0], arr.shape[0]), dtype=float)
    for i in range(arr.shape[0]):
        ret[i, :] = np.sqrt(np.sum((arr - arr[i, :])**2, axis=1))
    return ret


if __name__ == '__main__':
    print(distance_matrix())
