import math


def plotPixel(x1, y1, x2, y2, dx, dy, decide):
    pk = 2 * dy - dx
    points = []

    for i in range(0, dx + 1):
        if decide:

            points.append((y1, x1))

        else:

            points.append((x1, y1))

        if (x1 < x2):
            x1 = x1 + 1
        else:
            x1 = x1 - 1
        if (pk < 0):

            if (decide == 0):

                pk = pk + 2 * dy
            else:

                pk = pk + 2 * dy
        else:
            if (y1 < y2):
                y1 = y1 + 1
            else:
                y1 = y1 - 1

            pk = pk + 2 * dy - 2 * dx
    return points


def findPoints(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if (dx > dy):

        points = plotPixel(x1, y1, x2, y2, dx, dy, 0)


    else:

        points = plotPixel(y1, x1, y2, x2, dy, dx, 1)

    return points


if __name__ == '__main__':
    max_size = 15 - 1
    angle = math.tan(22.5 / 180 * math.pi)
    x1 = 7
    y1 = 14

    y2 = 0
    x2 = x1 + abs(y1) * angle
    print(x2)

    findPoints(x1, y1, x2, y2)

    x2 = max_size
    y2 = y1 - abs(x2 - x1) * angle
    findPoints(x1, y1, x2, y2)

    x2 = max_size
    y2 = y1 + abs(x2 - x1) * angle
    findPoints(x1, y1, x2, y2)

    y2 = max_size
    x2 = x1 + abs(y2 - y1) * angle

    findPoints(x1, y1, x2, y2)

    y2 = max_size
    x2 = x1 - abs(y2 - y1) * angle

    findPoints(x1, y1, x2, y2)

    x2 = 0
    y2 = y1 + abs(x1) * angle

    findPoints(x1, y1, x2, y2)

    x2 = 0
    y2 = y1 - abs(x1) * angle

    findPoints(x1, y1, x2, y2)

    y2 = 0
    x2 = x1 - abs(y1) * angle

    findPoints(x1, y1, x2, y2)

    print(100 * 1.1)
