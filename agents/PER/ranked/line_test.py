# Python3 program for Bresenhams Line Generation
import math


def plotPixel(x1, y1, x2, y2, dx, dy, decide):
    # pk is initial decision making parameter
    # Note:x1&y1,x2&y2, dx&dy values are interchanged
    # and passed in plotPixel function so
    # it can handle both cases when m>1 & m<1
    pk = 2 * dy - dx
    points = []

    # for (int i = 0; i <= dx; i++) {

    # if decide:
    #     print(dx, y1, x1, y2, x2)
    # else:
    #     print(dx, x1, y1, x2, y2)
    for i in range(0, dx + 1):
        if decide:
            # print(f"({y1},{x1})", end=" ")
            points.append((y1, x1))

        else:
            # print(f"({x1},{y1})", end=" ")
            points.append((x1, y1))

        # checking either to decrement or increment the
        # value if we have to plot from (0,100) to (100,0)
        if (x1 < x2):
            x1 = x1 + 1
        else:
            x1 = x1 - 1
        if (pk < 0):

            # decision value will decide to plot
            # either  x1 or y1 in x's position
            if (decide == 0):

                # putpixel(x1, y1, RED);
                pk = pk + 2 * dy
            else:

                # (y1,x1) is passed in xt
                # putpixel(y1, x1, YELLOW);
                pk = pk + 2 * dy
        else:
            if (y1 < y2):
                y1 = y1 + 1
            else:
                y1 = y1 - 1

            # if (decide == 0):
            #   # putpixel(x1, y1, RED)
            # else:
            #   #  putpixel(y1, x1, YELLOW);
            pk = pk + 2 * dy - 2 * dx
    return points


# Driver code

def findPoints(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    # print(dx,dy)
    # If slope is less than one
    if (dx > dy):
        # passing argument as 0 to plot(x,y)
        points = plotPixel(x1, y1, x2, y2, dx, dy, 0)

    # if slope is greater than or equal to 1
    else:
        # passing argument as 1 to plot (y,x)
        points = plotPixel(y1, x1, y2, x2, dy, dx, 1)
    # print(points)
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

    # x2 = x1 + abs(0-x1) * math.tan(25 / 180 * math.pi)
    # y2 = 0
    # findPoints(x1,y1,x2,y2)
    #
    #
    # x2 = x1 + abs(0-x1) * math.tan(25 / 180 * math.pi)
    # y2 = 0
    # findPoints(x1,y1,x2,y2)
    print(100 * 1.1)
