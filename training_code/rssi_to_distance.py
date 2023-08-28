import math
from pprint import pprint


class App():
    def __init__(self, rows, txpower):
        self.empty_rows = rows
        self.txpower = txpower

    def get_data_lines_from_file(self, path):
        rv = []
        with open(path) as f:
            lines = f.readlines()

        return lines

    def format_lines(self, lines):
        rv = []
        # columns are separated by whitespace, so remove it and place values in a list
        for i in range(len(lines)):
            lines[i] = ' '.join(lines[i].split()).split()

        # extracting columns names from the top
        column_names = lines[0]

        # removing rows with no data
        lines = lines[self.empty_rows - 1:]

        for line in lines:
            result = {}
            for i in range(len(line)):
                result.update({
                    column_names[i]: line[i],
                }
                )
            rv.append(result)
        return rv[1:]

    def calculate_distance(self, rssi):
        N = 2
        ratio = (self.txpower - int(rssi)) / (10 * N)
        distance = math.pow(10, ratio)
        return distance


if __name__ == '__main__':
    app = App(rows=2, txpower=-47)

    # data = app.get_data_lines_from_file('test.txt')
    # formatted_data = app.format_lines(data)

    try:

        # for i in range(len(formatted_data)):
        #     rssi = formatted_data[i]['RSSI']
        #     distance = app.calculate_distance(rssi)
        #     formatted_data[i].update({'distance': distance})

        rssi = -65
        distance = app.calculate_distance(rssi)
        print(distance, "m")

    except KeyError:
        print("No RSSI value in a file")

    # pprint(formatted_data)
